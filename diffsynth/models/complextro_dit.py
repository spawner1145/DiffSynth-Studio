import torch, math, functools
import torch.nn as nn
from typing import Tuple, Optional, Union, List
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from .general_modules import TimestepEmbeddings, RMSNorm, AdaLayerNorm
from ..core.gradient import gradient_checkpoint_forward

SEQ_MULTI_OF = 32

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False


def complextro_image_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, attention_mask = None, enable_fp8_attention: bool = False):
    if FLASH_ATTN_3_AVAILABLE and attention_mask is None:
        if not enable_fp8_attention:
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            x = flash_attn_interface.flash_attn_func(q, k, v)
            if isinstance(x, tuple):
                x = x[0]
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        else:
            origin_dtype = q.dtype
            q_std, k_std, v_std = q.std(), k.std(), v.std()
            q, k, v = (q / q_std).to(torch.float8_e4m3fn), (k / k_std).to(torch.float8_e4m3fn), (v / v_std).to(torch.float8_e4m3fn)
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            x = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=q_std * k_std / math.sqrt(q.size(-1)))
            if isinstance(x, tuple):
                x = x[0]
            x = x.to(origin_dtype) * v_std
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)

def apply_rotary_emb_complextro(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]
):
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if freqs_cis.ndim == x_rotated.ndim - 1:
        freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class ComplextroEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.rope_cache = {}
        self.scale_rope = scale_rope
        
    def rope_params(self, index, dim, theta=10000):
        """
            Args:
                index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs


    def _expand_pos_freqs_if_needed(self, video_fhw, txt_seq_lens):
        if isinstance(video_fhw, list):
            video_fhw = tuple(max([i[j] for i in video_fhw]) for j in range(3))
        _, height, width = video_fhw
        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)
        required_len = max_vid_index + max(txt_seq_lens)
        cur_max_len = self.pos_freqs.shape[0]
        if required_len <= cur_max_len:
            return

        new_max_len = math.ceil(required_len / 512) * 512
        pos_index = torch.arange(new_max_len)
        neg_index = torch.arange(new_max_len).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ], dim=1)
        return


    def forward(self, video_fhw, txt_seq_lens, device):
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone().contiguous()
            vid_freqs.append(self.rope_cache[rope_key])

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


    def forward_sampling(self, video_fhw, txt_seq_lens, device):
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if idx > 0 and f"{0}_{height}_{width}" not in self.rope_cache:
                frame_0, height_0, width_0 = video_fhw[0]

                rope_key_0 = f"0_{height_0}_{width_0}"
                spatial_freqs_0 = self.rope_cache[rope_key_0].reshape(frame_0, height_0, width_0, -1)
                h_indices = torch.linspace(0, height_0 - 1, height).long()
                w_indices = torch.linspace(0, width_0 - 1, width).long()
                h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
                sampled_rope = spatial_freqs_0[:, h_grid, w_grid, :]

                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                sampled_rope[:, :, :, :freqs_frame.shape[-1]] = freqs_frame

                seq_lens = frame * height * width
                self.rope_cache[rope_key] = sampled_rope.reshape(seq_lens, -1).clone()
            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone()
            vid_freqs.append(self.rope_cache[rope_key].contiguous())

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


class ComplextroEmbedLayer3DRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        video_fhw = [video_fhw]
        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx != layer_num:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            else:
                ### For the condition image, we set the layer index to -1
                video_freq = self._compute_condition_freqs(frame, height, width)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()

    @functools.lru_cache(maxsize=None)
    def _compute_condition_freqs(self, frame, height, width):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class ComplextroFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([])
        self.net.append(ApproximateGELU(dim, inner_dim))
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class ComplextroDoubleStreamAttention(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim_a, dim_a)
        self.to_k = nn.Linear(dim_a, dim_a)
        self.to_v = nn.Linear(dim_a, dim_a)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        self.add_q_proj = nn.Linear(dim_b, dim_b)
        self.add_k_proj = nn.Linear(dim_b, dim_b)
        self.add_v_proj = nn.Linear(dim_b, dim_b)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        self.to_out = torch.nn.Sequential(nn.Linear(dim_a, dim_a))
        self.to_add_out = nn.Linear(dim_b, dim_b)

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        enable_fp8_attention: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        img_q = rearrange(img_q, 'b s (h d) -> b h s d', h=self.num_heads)
        img_k = rearrange(img_k, 'b s (h d) -> b h s d', h=self.num_heads)
        img_v = rearrange(img_v, 'b s (h d) -> b h s d', h=self.num_heads)

        txt_q = rearrange(txt_q, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_k = rearrange(txt_k, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_v = rearrange(txt_v, 'b s (h d) -> b h s d', h=self.num_heads)

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)
        
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_complextro(img_q, img_freqs)
            img_k = apply_rotary_emb_complextro(img_k, img_freqs)
            txt_q = apply_rotary_emb_complextro(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_complextro(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        joint_attn_out = complextro_image_flash_attention(joint_q, joint_k, joint_v, num_heads=joint_q.shape[1], attention_mask=attention_mask, enable_fp8_attention=enable_fp8_attention).to(joint_q.dtype)

        txt_attn_output = joint_attn_out[:, :seq_txt, :]
        img_attn_output = joint_attn_out[:, seq_txt:, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class ComplextroSingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        enable_fp8_attention: bool = False,
    ) -> torch.FloatTensor:
        q, k, v = self.to_q(hidden_states), self.to_k(hidden_states), self.to_v(hidden_states)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        q, k = self.norm_q(q), self.norm_k(k)

        if image_rotary_emb is not None:
            q = apply_rotary_emb_complextro(q, image_rotary_emb)
            k = apply_rotary_emb_complextro(k, image_rotary_emb)

        attn_out = complextro_image_flash_attention(
            q, k, v,
            num_heads=q.shape[1],
            attention_mask=attention_mask,
            enable_fp8_attention=enable_fp8_attention,
        ).to(q.dtype)
        attn_out = self.to_out(attn_out)
        return attn_out


class ComplextroImageTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_attention_heads: int, 
        attention_head_dim: int, 
        eps: float = 1e-6,
    ):    
        super().__init__()
        
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim), 
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = ComplextroDoubleStreamAttention(
            dim_a=dim,
            dim_b=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = ComplextroFeedForward(dim=dim, dim_out=dim)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True), 
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = ComplextroFeedForward(dim=dim, dim_out=dim)
    
    def _modulate(self, x, mod_params, index=None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        image: torch.Tensor,  
        text: torch.Tensor,
        temb: torch.Tensor, 
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        enable_fp8_attention = False,
        modulate_index: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each
        if modulate_index is not None:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn, index=modulate_index)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        img_attn_out, txt_attn_out = self.attn(
            image=img_modulated,
            text=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            enable_fp8_attention=enable_fp8_attention,
        )
        
        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp, index=modulate_index)

        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        img_mlp_out = self.img_mlp(img_modulated_2)
        txt_mlp_out = self.txt_mlp(txt_modulated_2)

        image = image + img_gate_2 * img_mlp_out
        text = text + txt_gate_2 * txt_mlp_out

        return text, image


class ComplextroSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        modulation: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.modulation = modulation

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = ComplextroSingleStreamAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.mlp = ComplextroFeedForward(dim=dim, dim_out=dim)

        if modulation:
            self.modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        enable_fp8_attention: bool = False,
    ) -> torch.Tensor:
        if self.modulation:
            if temb is None:
                raise ValueError("temb must be provided when modulation is enabled.")
            mod = self.modulation_mlp(temb)
            if mod.ndim == 2:
                mod = mod.unsqueeze(1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

            normed = self.norm1(hidden_states)
            normed = normed * (1 + scale_msa) + shift_msa
            attn_out = self.attn(
                hidden_states=normed,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                enable_fp8_attention=enable_fp8_attention,
            )
            hidden_states = hidden_states + gate_msa * attn_out

            normed = self.norm2(hidden_states)
            normed = normed * (1 + scale_mlp) + shift_mlp
            mlp_out = self.mlp(normed)
            hidden_states = hidden_states + gate_mlp * mlp_out
        else:
            attn_out = self.attn(
                hidden_states=self.norm1(hidden_states),
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                enable_fp8_attention=enable_fp8_attention,
            )
            hidden_states = hidden_states + attn_out
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return hidden_states


class ComplextroImageDiT(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
        num_refiner_layers: int = 2,
        use_layer3d_rope: bool = False,
        use_additional_t_cond: bool = False,
        in_channels: int = 128,
        text_embed_dim: int = 1024,
        siglip_feat_dim: Optional[int] = None,
        hidden_size: int = 3072,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        rope_axes_dim: Optional[List[int]] = None,
    ):
        super().__init__()

        if hidden_size != num_attention_heads * attention_head_dim:
            raise ValueError(
                f"hidden_size({hidden_size}) must equal num_attention_heads({num_attention_heads}) * attention_head_dim({attention_head_dim})."
            )

        if rope_axes_dim is None:
            rope_axes_dim = [16, 56, 56]
        if sum(rope_axes_dim) != attention_head_dim:
            raise ValueError(
                f"sum(rope_axes_dim)={sum(rope_axes_dim)} must equal attention_head_dim={attention_head_dim}."
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.rope_axes_dim = rope_axes_dim

        if not use_layer3d_rope:
            self.pos_embed = ComplextroEmbedRope(theta=10000, axes_dim=self.rope_axes_dim, scale_rope=True)
        else:
            self.pos_embed = ComplextroEmbedLayer3DRope(theta=10000, axes_dim=self.rope_axes_dim, scale_rope=True)

        self.time_text_embed = TimestepEmbeddings(
            256,
            self.hidden_size,
            diffusers_compatible_format=True,
            scale=1000,
            align_dtype_to_timestep=False,
            use_additional_t_cond=use_additional_t_cond,
        )
        self.txt_norm = RMSNorm(text_embed_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.hidden_size)
        self.txt_in = nn.Linear(text_embed_dim, self.hidden_size)

        self.noise_refiner = nn.ModuleList(
            [
                ComplextroSingleTransformerBlock(
                    dim=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ComplextroSingleTransformerBlock(
                    dim=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    modulation=False,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.siglip_feat_dim = siglip_feat_dim
        if siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                RMSNorm(siglip_feat_dim, eps=1e-6),
                nn.Linear(siglip_feat_dim, self.hidden_size),
            )
            self.siglip_refiner = nn.ModuleList(
                [
                    ComplextroSingleTransformerBlock(
                        dim=self.hidden_size,
                        num_attention_heads=self.num_attention_heads,
                        attention_head_dim=self.attention_head_dim,
                        modulation=False,
                    )
                    for _ in range(num_refiner_layers)
                ]
            )
        else:
            self.siglip_embedder = None
            self.siglip_refiner = None

        self.transformer_blocks = nn.ModuleList(
            [
                ComplextroSingleTransformerBlock(
                    dim=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    modulation=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(self.hidden_size, single=True)
        self.proj_out = nn.Linear(self.hidden_size, in_channels)

    @staticmethod
    def _pad_tokens(tokens: torch.Tensor, pad_multiple: int = SEQ_MULTI_OF) -> Tuple[torch.Tensor, int, int]:
        ori_len = tokens.shape[0]
        pad_len = (-ori_len) % pad_multiple
        if pad_len > 0:
            tokens = torch.cat([tokens, tokens[-1:].repeat(pad_len, 1)], dim=0)
        return tokens, ori_len, ori_len + pad_len

    @staticmethod
    def _flatten_latent(latent: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if latent.ndim == 4:
            if latent.shape[0] != 1:
                raise ValueError("Omni mode expects per-image latent with batch=1.")
            latent = latent.squeeze(0)
        if latent.ndim != 3:
            raise ValueError("Latent must be (C,H,W) or (1,C,H,W) in omni mode.")
        _, height, width = latent.shape
        tokens = rearrange(latent, "C H W -> (H W) C")
        return tokens, (height, width)

    @staticmethod
    def _flatten_siglip(siglip: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if siglip.ndim == 4:
            if siglip.shape[0] != 1:
                raise ValueError("SigLIP feats in omni mode should have batch=1.")
            siglip = siglip.squeeze(0)
        if siglip.ndim == 3:
            if siglip.shape[-1] <= 0:
                raise ValueError("SigLIP feats must have channel dimension.")
            height, width, _ = siglip.shape
            tokens = siglip.reshape(height * width, siglip.shape[-1])
            return tokens, (height, width)
        if siglip.ndim == 2:
            seq_len = siglip.shape[0]
            side = int(math.sqrt(seq_len))
            if side * side != seq_len:
                raise ValueError("SigLIP token sequence is not square; provide (H,W,C).")
            return siglip, (side, side)
        raise ValueError("SigLIP feats must be (H,W,C), (S,C), or (1,H,W,C) in omni mode.")

    def _unpatchify_omni(
        self,
        unified_tokens: torch.Tensor,
        sizes: List[List[Optional[Tuple[int, int]]]],
        lengths: List[List[int]],
        x_pos_offsets: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        result = []
        for i, tokens in enumerate(unified_tokens):
            start, end = x_pos_offsets[i]
            unified_x = tokens[start:end]
            offset = 0
            out = None
            for size, total_len in zip(sizes[i], lengths[i]):
                if size is None:
                    offset += total_len
                    continue
                height, width = size
                ori_len = height * width
                local = unified_x[offset : offset + ori_len]
                out = rearrange(local, "(H W) C -> C H W", H=height, W=width)
                offset += total_len
            result.append(out)
        return result

    @staticmethod
    def _build_padded_unified(
        unified_list: List[torch.Tensor],
        freqs_list: List[torch.Tensor],
        seq_lens: List[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(seq_lens)
        unified = pad_sequence(unified_list, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(freqs_list, batch_first=True, padding_value=0.0)

        key_mask = torch.zeros((len(seq_lens), 1, 1, max_len), device=device, dtype=dtype)
        for i, seq_len in enumerate(seq_lens):
            if seq_len < max_len:
                key_mask[i, 0, 0, seq_len:] = float("-inf")

        return unified, unified_freqs, key_mask

    @staticmethod
    def _resolve_omni_image_noise_flags(
        latent_items: List[Optional[torch.Tensor]],
        image_noise_mask_item: Optional[List[int]],
    ) -> List[int]:
        if image_noise_mask_item is not None:
            flags = [int(v) for v in image_noise_mask_item]
            if len(flags) < len(latent_items):
                flags = flags + [flags[-1] if len(flags) > 0 else 0] * (len(latent_items) - len(flags))
            return flags[: len(latent_items)]

        flags = [0] * len(latent_items)
        target_idx = None
        for idx in range(len(latent_items) - 1, -1, -1):
            if latent_items[idx] is not None:
                target_idx = idx
                break
        if target_idx is None and len(latent_items) > 0:
            target_idx = len(latent_items) - 1
        if target_idx is not None:
            flags[target_idx] = 1
        return flags

    @staticmethod
    def _build_per_token_temb(
        token_noise_mask: List[int],
        temb_noisy: torch.Tensor,
        temb_clean: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.tensor(token_noise_mask, dtype=torch.long, device=temb_noisy.device)
        return torch.where(
            mask.view(1, -1, 1) == 1,
            temb_noisy.unsqueeze(1),
            temb_clean.unsqueeze(1),
        )

    def _build_omni_image_freqs(
        self,
        image_sizes: List[Optional[Tuple[int, int]]],
        image_lengths: List[int],
        txt_seq_len: int,
        device: torch.device,
        siglip_sizes: Optional[List[Optional[Tuple[int, int]]]] = None,
        siglip_lengths: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        real_shapes: List[Tuple[int, int, int]] = []
        for size in image_sizes:
            if size is not None:
                height, width = size
                real_shapes.append((1, height, width))
        if siglip_sizes is not None:
            for size in siglip_sizes:
                if size is not None:
                    height, width = size
                    real_shapes.append((1, height, width))

        if len(real_shapes) == 0:
            real_shapes = [(1, 1, 1)]

        all_img_freqs, txt_freqs = self.pos_embed(real_shapes, [txt_seq_len], device=device)
        txt_freqs = txt_freqs[:txt_seq_len]

        template_freq = all_img_freqs[:1]

        cursor = 0
        segment_freqs = []

        for size, total_len in zip(image_sizes, image_lengths):
            if size is None:
                segment_freqs.append(template_freq.repeat(total_len, 1))
                continue

            height, width = size
            ori_len = height * width
            local = all_img_freqs[cursor : cursor + ori_len]
            cursor += ori_len
            if total_len > local.shape[0]:
                local = torch.cat([local, local[-1:].repeat(total_len - local.shape[0], 1)], dim=0)
            segment_freqs.append(local[:total_len])

        if siglip_sizes is not None and siglip_lengths is not None:
            for size, total_len in zip(siglip_sizes, siglip_lengths):
                if size is None:
                    segment_freqs.append(template_freq.repeat(total_len, 1))
                    continue

                height, width = size
                ori_len = height * width
                local = all_img_freqs[cursor : cursor + ori_len]
                cursor += ori_len
                if total_len > local.shape[0]:
                    local = torch.cat([local, local[-1:].repeat(total_len - local.shape[0], 1)], dim=0)
                segment_freqs.append(local[:total_len])

        img_freqs = torch.cat(segment_freqs, dim=0)
        return img_freqs, txt_freqs


    def process_entity_masks(self, latents, prompt_emb, prompt_emb_mask, entity_prompt_emb, entity_prompt_emb_mask, entity_masks, height, width, image, img_shapes):
        # prompt_emb
        all_prompt_emb = entity_prompt_emb + [prompt_emb]
        all_prompt_emb = [self.txt_in(self.txt_norm(local_prompt_emb)) for local_prompt_emb in all_prompt_emb]
        all_prompt_emb = torch.cat(all_prompt_emb, dim=1)

        # image_rotary_emb
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        entity_seq_lens = [emb_mask.sum(dim=1).tolist() for emb_mask in entity_prompt_emb_mask]
        entity_rotary_emb = [self.pos_embed(img_shapes, entity_seq_len, device=latents.device)[1] for entity_seq_len in entity_seq_lens]
        txt_rotary_emb = torch.cat(entity_rotary_emb + [image_rotary_emb[1]], dim=0)
        image_rotary_emb = (image_rotary_emb[0], txt_rotary_emb)

        # attention_mask
        repeat_dim = latents.shape[1]
        max_masks = entity_masks.shape[1]
        entity_masks = entity_masks.repeat(1, 1, repeat_dim, 1, 1)
        entity_masks = [entity_masks[:, i, None].squeeze(1) for i in range(max_masks)]
        global_mask = torch.ones_like(entity_masks[0]).to(device=latents.device, dtype=latents.dtype)
        entity_masks = entity_masks + [global_mask]

        N = len(entity_masks)
        batch_size = entity_masks[0].shape[0]
        seq_lens = [emb_mask.sum(dim=1).item() for emb_mask in entity_prompt_emb_mask] + [prompt_emb_mask.sum(dim=1).item()]
        text_seq_len = sum(seq_lens)
        total_seq_len = text_seq_len + image.shape[1]
        patched_masks = []
        for i in range(N):
            patched_mask = rearrange(entity_masks[i], "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
            patched_masks.append(patched_mask)
        attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), dtype=torch.bool).to(device=entity_masks[0].device)

        # prompt-image attention mask
        image_start = text_seq_len
        image_end = total_seq_len
        cumsum = [0]
        single_image_seq = image_end - image_start
        for length in seq_lens:
            cumsum.append(cumsum[-1] + length)
        for i in range(N):
            prompt_start = cumsum[i]
            prompt_end = cumsum[i+1]
            image_mask = torch.sum(patched_masks[i], dim=-1) > 0
            image_mask = image_mask.unsqueeze(1).repeat(1, seq_lens[i], 1)
            # repeat image mask to match the single image sequence length
            repeat_time = single_image_seq // image_mask.shape[-1]
            image_mask = image_mask.repeat(1, 1, repeat_time)
            # prompt update with image
            attention_mask[:, prompt_start:prompt_end, image_start:image_end] = image_mask
            # image update with prompt
            attention_mask[:, image_start:image_end, prompt_start:prompt_end] = image_mask.transpose(1, 2)
        # prompt-prompt attention mask, let the prompt tokens not attend to each other
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                start_i, end_i = cumsum[i], cumsum[i+1]
                start_j, end_j = cumsum[j], cumsum[j+1]
                attention_mask[:, start_i:end_i, start_j:end_j] = False

        prompt_valid_mask = torch.cat(entity_prompt_emb_mask + [prompt_emb_mask], dim=1).bool()
        prompt_to_prompt = prompt_valid_mask.unsqueeze(1) & prompt_valid_mask.unsqueeze(2)
        attention_mask[:, :text_seq_len, :text_seq_len] = (
            attention_mask[:, :text_seq_len, :text_seq_len] & prompt_to_prompt
        )
        attention_mask[:, :text_seq_len, image_start:image_end] = (
            attention_mask[:, :text_seq_len, image_start:image_end] & prompt_valid_mask.unsqueeze(-1)
        )
        attention_mask[:, image_start:image_end, :text_seq_len] = (
            attention_mask[:, image_start:image_end, :text_seq_len] & prompt_valid_mask.unsqueeze(1)
        )

        attention_mask = attention_mask.float()
        attention_mask[attention_mask == 0] = float('-inf')
        attention_mask[attention_mask == 1] = 0
        attention_mask = attention_mask.to(device=latents.device, dtype=latents.dtype).unsqueeze(1)

        return all_prompt_emb, image_rotary_emb, attention_mask


    def forward(
        self,
        latents=None,
        timestep=None,
        prompt_emb=None,
        prompt_emb_mask=None,
        height=None,
        width=None,
        siglip_feats: Optional[torch.Tensor] = None,
        image_noise_mask: Optional[List[List[int]]] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        if latents is None:
            raise ValueError("latents must be provided.")

        omni_mode = isinstance(latents, list) and len(latents) > 0 and isinstance(latents[0], list)

        if omni_mode:
            if siglip_feats is not None and (not isinstance(siglip_feats, list) or not isinstance(siglip_feats[0], list)):
                raise ValueError("siglip_feats must be List[List[Tensor]] when latents is omni-mode.")
            if image_noise_mask is not None and (not isinstance(image_noise_mask, list) or not isinstance(image_noise_mask[0], list)):
                raise ValueError("image_noise_mask must be List[List[int]] when latents is omni-mode.")

            batch_size = len(latents)
            text_tokens = self.txt_in(self.txt_norm(prompt_emb))
            conditioning_noisy = self.time_text_embed(timestep, text_tokens.dtype)
            conditioning_clean = self.time_text_embed(torch.ones_like(timestep), text_tokens.dtype)

            unified_list = []
            freqs_list = []
            temb_list = []
            seq_lens = []
            x_sizes = []
            x_lengths = []
            x_pos_offsets = []

            for b in range(batch_size):
                noise_flags = self._resolve_omni_image_noise_flags(
                    latents[b],
                    image_noise_mask[b] if image_noise_mask is not None and b < len(image_noise_mask) else None,
                )

                image_tokens_list = []
                size_list = []
                length_list = []
                image_token_noise_mask = []
                for img_idx, img in enumerate(latents[b]):
                    local_noise_flag = noise_flags[img_idx]
                    if img is None:
                        pad_len = SEQ_MULTI_OF
                        image_tokens_list.append(torch.zeros((pad_len, self.img_in.in_features), device=prompt_emb.device, dtype=prompt_emb.dtype))
                        size_list.append(None)
                        length_list.append(pad_len)
                        image_token_noise_mask.extend([local_noise_flag] * pad_len)
                        continue
                    tokens, (h, w) = self._flatten_latent(img)
                    tokens, _, total_len = self._pad_tokens(tokens)
                    image_tokens_list.append(tokens)
                    size_list.append((h, w))
                    length_list.append(total_len)
                    image_token_noise_mask.extend([local_noise_flag] * total_len)

                image_tokens = torch.cat(image_tokens_list, dim=0)
                image_tokens = self.img_in(image_tokens)
                image_temb = self._build_per_token_temb(
                    image_token_noise_mask,
                    conditioning_noisy[b : b + 1],
                    conditioning_clean[b : b + 1],
                )

                for block in self.noise_refiner:
                    image_tokens = gradient_checkpoint_forward(
                        block,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                        hidden_states=image_tokens.unsqueeze(0),
                        temb=image_temb,
                    ).squeeze(0)

                txt_seq_len = (
                    int(prompt_emb_mask[b].sum().item())
                    if prompt_emb_mask is not None
                    else text_tokens.shape[1]
                )
                text_tokens_b = text_tokens[b : b + 1, :txt_seq_len]
                for block in self.context_refiner:
                    text_tokens_b = gradient_checkpoint_forward(
                        block,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                        hidden_states=text_tokens_b,
                    )

                sig_tokens = None
                sig_shapes = []
                sig_lengths = []
                sig_token_noise_mask = []
                if siglip_feats is not None:
                    if self.siglip_embedder is None:
                        raise ValueError("siglip_feats provided but siglip_feat_dim is None.")
                    sig_list = []
                    for sig_idx, sig in enumerate(siglip_feats[b]):
                        local_noise_flag = noise_flags[sig_idx] if sig_idx < len(noise_flags) else noise_flags[-1]
                        if sig is None:
                            pad_len = SEQ_MULTI_OF
                            sig_list.append(torch.zeros((pad_len, self.siglip_feat_dim), device=prompt_emb.device, dtype=prompt_emb.dtype))
                            sig_shapes.append(None)
                            sig_lengths.append(pad_len)
                            sig_token_noise_mask.extend([local_noise_flag] * pad_len)
                            continue
                        sig_tok, (sh, sw) = self._flatten_siglip(sig)
                        sig_tok, _, sig_total_len = self._pad_tokens(sig_tok)
                        sig_list.append(sig_tok)
                        sig_shapes.append((sh, sw))
                        sig_lengths.append(sig_total_len)
                        sig_token_noise_mask.extend([local_noise_flag] * sig_total_len)
                    sig_tokens = torch.cat(sig_list, dim=0)
                    sig_tokens = self.siglip_embedder(sig_tokens)
                    for block in self.siglip_refiner:
                        sig_tokens = gradient_checkpoint_forward(
                            block,
                            use_gradient_checkpointing=use_gradient_checkpointing,
                            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                            hidden_states=sig_tokens.unsqueeze(0),
                        ).squeeze(0)

                txt_seq_lens = [txt_seq_len]
                img_freqs, txt_freqs = self._build_omni_image_freqs(
                    image_sizes=size_list,
                    image_lengths=length_list,
                    txt_seq_len=txt_seq_len,
                    device=prompt_emb.device,
                    siglip_sizes=sig_shapes if sig_tokens is not None else None,
                    siglip_lengths=sig_lengths if sig_tokens is not None else None,
                )

                txt_token_noise_mask = [0] * txt_seq_len
                unified_token_noise_mask = txt_token_noise_mask + image_token_noise_mask + sig_token_noise_mask
                unified_temb = self._build_per_token_temb(
                    unified_token_noise_mask,
                    conditioning_noisy[b : b + 1],
                    conditioning_clean[b : b + 1],
                ).squeeze(0)

                # Align omni unified order with z_image_dit: [cap, x, siglip]
                unified = torch.cat([text_tokens_b.squeeze(0), image_tokens] + ([sig_tokens] if sig_tokens is not None else []), dim=0)
                unified_freqs = torch.cat([txt_freqs, img_freqs], dim=0)

                unified_list.append(unified)
                freqs_list.append(unified_freqs)
                temb_list.append(unified_temb)
                seq_lens.append(unified.shape[0])
                x_sizes.append(size_list)
                x_lengths.append(length_list)
                x_pos_offsets.append((txt_seq_len, txt_seq_len + sum(length_list)))

            unified, unified_freqs, key_mask = self._build_padded_unified(
                unified_list,
                freqs_list,
                seq_lens,
                dtype=text_tokens.dtype,
                device=prompt_emb.device,
            )
            unified_temb = pad_sequence(temb_list, batch_first=True, padding_value=0.0)

            for block in self.transformer_blocks:
                unified = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                    hidden_states=unified,
                    temb=unified_temb,
                    image_rotary_emb=unified_freqs,
                    attention_mask=key_mask,
                )

            image_tokens = unified
            image_tokens = self.norm_out(image_tokens, conditioning_noisy)
            image_tokens = self.proj_out(image_tokens)

            outputs = self._unpatchify_omni(image_tokens, x_sizes, x_lengths, x_pos_offsets)
            outputs = [out.unsqueeze(0) for out in outputs]
            return torch.cat(outputs, dim=0)

        if latents.ndim == 4:
            latent_h, latent_w = latents.shape[-2:]
            image_tokens = rearrange(latents, "B C H W -> B (H W) C")
            return_latents_4d = True
        else:
            image_tokens = latents
            return_latents_4d = False
            if height is not None and width is not None:
                latent_h, latent_w = height // 16, width // 16
            else:
                seq_len = image_tokens.shape[1]
                side = int(math.sqrt(seq_len))
                if side * side != seq_len:
                    raise ValueError("height/width required when latents are tokenized and sequence is not square.")
                latent_h = latent_w = side

        text_tokens = self.txt_in(self.txt_norm(prompt_emb))
        image_tokens = self.img_in(image_tokens)

        conditioning = self.time_text_embed(timestep, image_tokens.dtype)

        # Refine image tokens (noise) and text tokens (context)
        for block in self.noise_refiner:
            image_tokens = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=image_tokens,
                temb=conditioning,
            )

        for block in self.context_refiner:
            text_tokens = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=text_tokens,
            )

        # SigLIP vision feats (optional)
        siglip_tokens = None
        siglip_shape = None
        if siglip_feats is not None:
            if self.siglip_embedder is None:
                raise ValueError("siglip_feats provided but siglip_feat_dim is None.")
            if siglip_feats.ndim == 4:
                siglip_h, siglip_w = siglip_feats.shape[1], siglip_feats.shape[2]
                siglip_tokens = siglip_feats.reshape(siglip_feats.shape[0], siglip_h * siglip_w, siglip_feats.shape[-1])
                siglip_shape = (1, siglip_h, siglip_w)
            elif siglip_feats.ndim == 3:
                seq_len = siglip_feats.shape[1]
                side = int(math.sqrt(seq_len))
                if side * side != seq_len:
                    raise ValueError("siglip_feats is tokenized but sequence is not square; provide 4D feats.")
                siglip_tokens = siglip_feats
                siglip_shape = (1, side, side)
            else:
                raise ValueError("siglip_feats must be 3D (B,S,C) or 4D (B,H,W,C).")

            siglip_tokens = self.siglip_embedder(siglip_tokens)
            for block in self.siglip_refiner:
                siglip_tokens = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                    hidden_states=siglip_tokens,
                )

        # RoPE for unified sequence (text + image + optional siglip)
        txt_seq_lens = (
            prompt_emb_mask.sum(dim=1).tolist()
            if prompt_emb_mask is not None
            else [text_tokens.shape[1]] * text_tokens.shape[0]
        )

        # Align basic unified order with z_image_dit: [x, cap]
        # (keep optional siglip support by appending after cap when present)
        unified_list = []
        freqs_list = []
        seq_lens = []
        x_seqlen = image_tokens.shape[1]
        x_pos_offsets = []

        for b in range(text_tokens.shape[0]):
            txt_len = int(txt_seq_lens[b])
            text_b = text_tokens[b, :txt_len]
            image_b = image_tokens[b]

            img_shapes = [(1, latent_h, latent_w)]
            if siglip_shape is not None:
                img_shapes.append(siglip_shape)

            img_freqs_all, txt_freqs = self.pos_embed(img_shapes, [txt_len], device=image_tokens.device)
            txt_freqs = txt_freqs[:txt_len]
            img_freqs = img_freqs_all[:x_seqlen]

            if siglip_tokens is not None:
                siglip_b = siglip_tokens[b]
                siglip_len = siglip_b.shape[0]
                siglip_freqs = img_freqs_all[x_seqlen : x_seqlen + siglip_len]
                unified_b = torch.cat([image_b, text_b, siglip_b], dim=0)
                freqs_b = torch.cat([img_freqs, txt_freqs, siglip_freqs], dim=0)
            else:
                unified_b = torch.cat([image_b, text_b], dim=0)
                freqs_b = torch.cat([img_freqs, txt_freqs], dim=0)

            unified_list.append(unified_b)
            freqs_list.append(freqs_b)
            seq_lens.append(unified_b.shape[0])
            x_pos_offsets.append((0, x_seqlen))

        unified, unified_freqs, key_mask = self._build_padded_unified(
            unified_list,
            freqs_list,
            seq_lens,
            dtype=image_tokens.dtype,
            device=image_tokens.device,
        )

        for block in self.transformer_blocks:
            unified = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=unified,
                temb=conditioning,
                image_rotary_emb=unified_freqs,
                attention_mask=key_mask,
            )

        image_tokens = []
        for b, (start, end) in enumerate(x_pos_offsets):
            image_tokens.append(unified[b, start:end, :])
        image_tokens = torch.stack(image_tokens, dim=0)
        image_tokens = self.norm_out(image_tokens, conditioning)
        image_tokens = self.proj_out(image_tokens)

        if return_latents_4d:
            latents = rearrange(image_tokens, "B (H W) C -> B C H W", H=latent_h, W=latent_w)
            return latents

        return image_tokens
