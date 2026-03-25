import torch, math, functools
import torch.nn as nn
from typing import Tuple, Optional, Union, List
from collections import OrderedDict
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from .general_modules import TimestepEmbeddings, RMSNorm, AdaLayerNorm
from ..core.gradient import gradient_checkpoint_forward

SEQ_MULTI_OF = 32
TOKEN_TYPE_TEXT = 0
TOKEN_TYPE_TARGET_IMAGE = 1
TOKEN_TYPE_SIGLIP = 2
TOKEN_TYPE_COND_IMAGE = 3

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False


class TreadRouter:
    def __init__(self):
        pass

    def get_mask(self, x, selection_rate=0.0):
        batch_size, num_patches, _ = x.shape
        device = x.device
        num_mask = int(num_patches * selection_rate)
        num_keep = max(1, num_patches - num_mask)
        noise_random = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise_random, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        return ids_keep

    def start_route(self, x, ids_keep):
        return x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, x.size(2)))

    def end_route(self, masked_x, ids_keep, original_x):
        return original_x.scatter(1, ids_keep.unsqueeze(-1).expand(-1, -1, original_x.size(2)), masked_x)


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
        self.rope_cache = OrderedDict()
        self.rope_cache_device = None
        self.max_rope_cache_entries = 32
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

    def _ensure_cache_device(self, device):
        device_key = str(device)
        if self.rope_cache_device != device_key:
            self.rope_cache.clear()
            self.rope_cache_device = device_key

    def _cache_get(self, key):
        cached = self.rope_cache.get(key)
        if cached is not None:
            self.rope_cache.move_to_end(key)
        return cached

    def _cache_set(self, key, value):
        self.rope_cache[key] = value
        self.rope_cache.move_to_end(key)
        while len(self.rope_cache) > self.max_rope_cache_entries:
            self.rope_cache.popitem(last=False)


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
        self._ensure_cache_device(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = (idx, frame, height, width)

            cached_freqs = self._cache_get(rope_key)
            if cached_freqs is None:
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
                cached_freqs = freqs.contiguous()
                self._cache_set(rope_key, cached_freqs)
            vid_freqs.append(cached_freqs)

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
        self._ensure_cache_device(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = (idx, frame, height, width)
            rope_key_0 = (0, video_fhw[0][0], video_fhw[0][1], video_fhw[0][2])
            cached_freqs = self._cache_get(rope_key)
            cached_freqs_0 = self._cache_get(rope_key_0)
            if idx > 0 and cached_freqs is None and cached_freqs_0 is not None:
                frame_0, height_0, width_0 = video_fhw[0]

                spatial_freqs_0 = cached_freqs_0.reshape(frame_0, height_0, width_0, -1)
                h_indices = torch.linspace(0, height_0 - 1, height).long()
                w_indices = torch.linspace(0, width_0 - 1, width).long()
                h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
                sampled_rope = spatial_freqs_0[:, h_grid, w_grid, :]

                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                sampled_rope[:, :, :, :freqs_frame.shape[-1]] = freqs_frame

                seq_lens = frame * height * width
                self._cache_set(rope_key, sampled_rope.reshape(seq_lens, -1).contiguous())
                cached_freqs = self._cache_get(rope_key)
            if cached_freqs is None:
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
                cached_freqs = freqs.contiguous()
                self._cache_set(rope_key, cached_freqs)
            vid_freqs.append(cached_freqs)

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
            self.modulation_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(dim, 6 * dim),
                    )
                    for _ in range(4)
                ]
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        enable_fp8_attention: bool = False,
    ) -> torch.Tensor:
        if self.modulation:
            if temb is None:
                raise ValueError("temb must be provided when modulation is enabled.")

            if token_type_ids is not None:
                target_batch = hidden_states.shape[0]

                if temb.ndim == 2:
                    if temb.shape[0] == 1 and target_batch > 1:
                        temb = temb.expand(target_batch, -1)
                    elif temb.shape[0] != target_batch:
                        raise ValueError("temb batch size must match hidden_states batch size.")
                    temb = temb.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
                elif temb.ndim == 3:
                    if temb.shape[0] == 1 and target_batch > 1:
                        temb = temb.expand(target_batch, -1, -1)
                    elif temb.shape[0] != target_batch:
                        raise ValueError("temb batch size must match hidden_states batch size.")
                    if temb.shape[1] == 1 and hidden_states.shape[1] > 1:
                        temb = temb.expand(-1, hidden_states.shape[1], -1)
                    elif temb.shape[1] != hidden_states.shape[1]:
                        raise ValueError("temb sequence length must match hidden_states when token_type_ids is provided.")
                else:
                    raise ValueError("temb must be 2D or 3D when token_type_ids is provided.")

                if token_type_ids.shape[0] == 1 and target_batch > 1:
                    token_type_ids = token_type_ids.expand(target_batch, -1)
                elif token_type_ids.shape[0] != target_batch:
                    raise ValueError("token_type_ids batch size must match hidden_states batch size.")
                if token_type_ids.shape[1] != hidden_states.shape[1]:
                    raise ValueError("token_type_ids sequence length must match hidden_states sequence length.")

                token_type_ids = token_type_ids.to(device=hidden_states.device, dtype=torch.long).clamp(
                    TOKEN_TYPE_TEXT, TOKEN_TYPE_COND_IMAGE
                )
                flat_temb = temb.reshape(-1, temb.shape[-1])
                flat_token_type_ids = token_type_ids.reshape(-1)
                flat_mod = torch.empty(
                    (flat_temb.shape[0], 6 * self.dim),
                    device=flat_temb.device,
                    dtype=flat_temb.dtype,
                )
                for type_id, mlp in enumerate(self.modulation_mlps):
                    type_mask = flat_token_type_ids == type_id
                    if type_mask.any():
                        flat_mod[type_mask] = mlp(flat_temb[type_mask])
                mod = flat_mod.view(*temb.shape[:-1], -1)
            else:
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
        latent_downsample_factor: Optional[int] = None,
        latent_patch_size: Optional[int] = None,
        text_embed_dim: Optional[int] = None,
        siglip_feat_dim: Optional[int] = None,
        hidden_size: int = 3072,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        rope_axes_dim: Optional[List[int]] = None,
        use_unified_token_type_modulation: bool = False,
        use_omni_token_type_modulation: bool = False,
        use_token_type_embedding: bool = True,
        enable_tread_routing: bool = False,
        tread_routes: Optional[List[dict]] = None,
        use_text_modulation: bool = False,
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
        if text_embed_dim is None:
            raise ValueError(
                "text_embed_dim must be provided for ComplextroImageDiT. "
                "Set it from text_encoder.model.config.text_config.hidden_size."
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.rope_axes_dim = rope_axes_dim
        if latent_downsample_factor is None:
            if int(in_channels) == 128:
                latent_downsample_factor = 16
            elif int(in_channels) == 16:
                latent_downsample_factor = 8
            else:
                raise ValueError(
                    "latent_downsample_factor must be provided when in_channels is not a known Complextro VAE latent "
                    f"spec (got in_channels={in_channels})."
                )
        if latent_patch_size is None:
            latent_patch_size = 2 if int(in_channels) == 16 else 1
        self.latent_downsample_factor = int(latent_downsample_factor)
        self.latent_patch_size = int(latent_patch_size)
        self.latent_channels = int(in_channels)
        self.use_unified_token_type_modulation = bool(use_unified_token_type_modulation or use_text_modulation)
        self.use_omni_token_type_modulation = bool(use_omni_token_type_modulation or use_text_modulation)
        self.use_token_type_embedding = bool(use_token_type_embedding)
        self.enable_tread_routing = bool(enable_tread_routing)
        self.use_text_modulation = bool(use_text_modulation)
        self.tread_routes = self._normalize_tread_routes(tread_routes)
        self.tread_router = TreadRouter() if self.enable_tread_routing and len(self.tread_routes) > 0 else None
        self._freq_cache = OrderedDict()
        self._scaled_siglip_freq_cache = OrderedDict()
        self._freq_cache_device = None
        self._max_freq_cache_entries = 32

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

        self.img_token_dim = int(in_channels) * self.latent_patch_size * self.latent_patch_size
        self.img_in = nn.Linear(self.img_token_dim, self.hidden_size)
        self.txt_in = nn.Linear(text_embed_dim, self.hidden_size)
        self.text_pool_proj = nn.Linear(text_embed_dim, self.hidden_size) if self.use_text_modulation else None
        self.edit_pool_proj = nn.Linear(self.hidden_size, self.hidden_size) if self.use_text_modulation else None
        self.image_pad_token = nn.Parameter(torch.empty((1, self.img_token_dim)))
        self.token_type_embed = nn.Embedding(4, self.hidden_size) if self.use_token_type_embedding else None

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
                    modulation=self.use_text_modulation,
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
            self.siglip_pad_token = nn.Parameter(torch.empty((1, self.hidden_size)))
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
            self.siglip_pad_token = None

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
        for route in self.tread_routes:
            if route["start_layer_idx"] >= num_layers or route["end_layer_idx"] >= num_layers:
                raise ValueError(
                    f"tread route layer idx out of range: start={route['start_layer_idx']}, end={route['end_layer_idx']}, num_layers={num_layers}."
                )
        self.norm_out = AdaLayerNorm(self.hidden_size, single=True)
        self.proj_out = nn.Linear(self.hidden_size, self.img_token_dim)

        # Initialize all Linear layers with xavier_uniform:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pad tokens (analogous to label embedding):
        nn.init.normal_(self.image_pad_token, mean=0.0, std=0.02)
        if self.siglip_pad_token is not None:
            nn.init.normal_(self.siglip_pad_token, mean=0.0, std=0.02)
        if self.token_type_embed is not None:
            nn.init.normal_(self.token_type_embed.weight, mean=0.0, std=0.02)

        for module in self.time_text_embed.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for block in list(self.transformer_blocks) + list(self.noise_refiner) + list(self.context_refiner):
            if block.modulation:
                # DiT-style adaLN-Zero: start each modulated block close to identity.
                nn.init.zeros_(block.modulation_mlp[-1].weight)
                nn.init.zeros_(block.modulation_mlp[-1].bias)
                for mlp in block.modulation_mlps:
                    nn.init.zeros_(mlp[-1].weight)
                    nn.init.zeros_(mlp[-1].bias)

        nn.init.zeros_(self.norm_out.linear.weight)
        nn.init.zeros_(self.norm_out.linear.bias)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    def _compute_text_pool_conditioning(self, prompt_emb: torch.Tensor, prompt_emb_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if prompt_emb_mask is not None:
            mask = prompt_emb_mask.unsqueeze(-1).to(dtype=prompt_emb.dtype)
            pooled = (prompt_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = prompt_emb.mean(dim=1)
        return self.text_pool_proj(pooled)

    def _compute_edit_pool_conditioning(
        self,
        image_tokens: torch.Tensor,
        cond_token_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.edit_pool_proj is None:
            return None
        cond_token_mask = cond_token_mask.to(device=image_tokens.device, dtype=image_tokens.dtype)
        if float(cond_token_mask.sum().item()) <= 0.0:
            return None
        pooled = (image_tokens * cond_token_mask.unsqueeze(-1)).sum(dim=0, keepdim=True)
        pooled = pooled / cond_token_mask.sum().clamp(min=1.0)
        return self.edit_pool_proj(pooled)

    @staticmethod
    def _build_condition_token_mask(
        length_list: List[int],
        valid_token_mask: List[int],
    ) -> torch.Tensor:
        cond_token_mask = []
        offset = 0
        for image_idx, image_len in enumerate(length_list):
            is_cond_image = image_idx != len(length_list) - 1
            local_valid = valid_token_mask[offset : offset + image_len]
            if is_cond_image:
                cond_token_mask.extend(local_valid)
            else:
                cond_token_mask.extend([0] * image_len)
            offset += image_len
        return torch.tensor(cond_token_mask, dtype=torch.float32)

    def _resolve_token_type_embed_ids(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        token_type_ids = token_type_ids.clamp(TOKEN_TYPE_TEXT, TOKEN_TYPE_COND_IMAGE)
        # Keep text and SigLIP distinct, but let cond/target images share the same role embedding.
        return torch.where(
            token_type_ids == TOKEN_TYPE_COND_IMAGE,
            torch.full_like(token_type_ids, TOKEN_TYPE_TARGET_IMAGE),
            token_type_ids,
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "image_pad_token" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["image_pad_token"] = self.image_pad_token.detach().clone()
        if self.token_type_embed is not None and "token_type_embed.weight" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["token_type_embed.weight"] = self.token_type_embed.weight.detach().clone()
        if self.text_pool_proj is not None and "text_pool_proj.weight" not in state_dict:
            if not isinstance(state_dict, dict):
                state_dict = dict(state_dict)
            state_dict["text_pool_proj.weight"] = self.text_pool_proj.weight.detach().clone()
            if self.text_pool_proj.bias is not None:
                state_dict["text_pool_proj.bias"] = self.text_pool_proj.bias.detach().clone()
        if self.edit_pool_proj is not None and "edit_pool_proj.weight" not in state_dict:
            if not isinstance(state_dict, dict):
                state_dict = dict(state_dict)
            state_dict["edit_pool_proj.weight"] = self.edit_pool_proj.weight.detach().clone()
            if self.edit_pool_proj.bias is not None:
                state_dict["edit_pool_proj.bias"] = self.edit_pool_proj.bias.detach().clone()

        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)

        for module_prefix, module in self.named_modules():
            if not isinstance(module, ComplextroSingleTransformerBlock) or not getattr(module, "modulation", False):
                continue

            base_prefix = f"{module_prefix}.modulation_mlp.1"
            base_weight_key = f"{base_prefix}.weight"
            base_bias_key = f"{base_prefix}.bias"

            if base_weight_key not in state_dict or base_bias_key not in state_dict:
                continue

            for head_idx in range(4):
                head_prefix = f"{module_prefix}.modulation_mlps.{head_idx}.1"
                head_weight_key = f"{head_prefix}.weight"
                head_bias_key = f"{head_prefix}.bias"
                if head_weight_key not in state_dict:
                    if head_idx == TOKEN_TYPE_COND_IMAGE:
                        old_image_weight_key = f"{module_prefix}.modulation_mlps.{TOKEN_TYPE_TARGET_IMAGE}.1.weight"
                        if old_image_weight_key in state_dict:
                            state_dict[head_weight_key] = state_dict[old_image_weight_key].detach().clone()
                        else:
                            state_dict[head_weight_key] = state_dict[base_weight_key].detach().clone()
                    else:
                        state_dict[head_weight_key] = state_dict[base_weight_key].detach().clone()
                if head_bias_key not in state_dict:
                    if head_idx == TOKEN_TYPE_COND_IMAGE:
                        old_image_bias_key = f"{module_prefix}.modulation_mlps.{TOKEN_TYPE_TARGET_IMAGE}.1.bias"
                        if old_image_bias_key in state_dict:
                            state_dict[head_bias_key] = state_dict[old_image_bias_key].detach().clone()
                        else:
                            state_dict[head_bias_key] = state_dict[base_bias_key].detach().clone()
                    else:
                        state_dict[head_bias_key] = state_dict[base_bias_key].detach().clone()

        if self.siglip_pad_token is not None:
            if "siglip_pad_token" not in state_dict:
                if not isinstance(state_dict, dict):
                    state_dict = dict(state_dict)
                state_dict["siglip_pad_token"] = self.siglip_pad_token.detach().clone()
            else:
                loaded_siglip_pad = state_dict["siglip_pad_token"]
                if tuple(loaded_siglip_pad.shape) != tuple(self.siglip_pad_token.shape):
                    if not isinstance(state_dict, dict):
                        state_dict = dict(state_dict)
                    state_dict["siglip_pad_token"] = self.siglip_pad_token.detach().clone()

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _normalize_tread_routes(routes: Optional[List[dict]]) -> List[dict]:
        if routes is None:
            return []
        if not isinstance(routes, list):
            raise ValueError("tread_routes must be a list of route configs.")

        normalized = []
        for route in routes:
            if not isinstance(route, dict):
                raise ValueError("Each tread route must be a dict.")
            if "start_layer_idx" not in route or "end_layer_idx" not in route:
                raise ValueError("Each tread route must include start_layer_idx and end_layer_idx.")
            selection_ratio = float(route.get("selection_ratio", 0.0))
            if not (0.0 <= selection_ratio < 1.0):
                raise ValueError("tread route selection_ratio must be in [0, 1).")
            start_layer_idx = int(route["start_layer_idx"])
            end_layer_idx = int(route["end_layer_idx"])
            if end_layer_idx < start_layer_idx:
                raise ValueError("tread route end_layer_idx must be >= start_layer_idx.")
            normalized.append(
                {
                    "selection_ratio": selection_ratio,
                    "start_layer_idx": start_layer_idx,
                    "end_layer_idx": end_layer_idx,
                }
            )

        normalized = sorted(normalized, key=lambda x: x["start_layer_idx"])
        for i in range(1, len(normalized)):
            if normalized[i]["start_layer_idx"] <= normalized[i - 1]["end_layer_idx"]:
                raise ValueError("tread routes must not overlap.")
        return normalized

    @staticmethod
    def _pad_tokens(
        tokens: torch.Tensor,
        pad_multiple: int = SEQ_MULTI_OF,
        pad_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        ori_len = tokens.shape[0]
        pad_len = (-ori_len) % pad_multiple
        if pad_len > 0:
            if pad_token is None:
                pad_values = tokens[-1:].repeat(pad_len, 1)
            else:
                pad_values = pad_token.to(device=tokens.device, dtype=tokens.dtype).repeat(pad_len, 1)
            tokens = torch.cat([tokens, pad_values], dim=0)
        return tokens, ori_len, ori_len + pad_len

    def _patchify_latent(self, latent: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if latent.ndim == 4:
            if latent.shape[0] != 1:
                raise ValueError("Omni mode expects per-image latent with batch=1.")
            latent = latent.squeeze(0)
        if latent.ndim != 3:
            raise ValueError("Latent must be (C,H,W) or (1,C,H,W) in omni mode.")
        _, height, width = latent.shape
        patch = self.latent_patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"Latent spatial shape {(height, width)} must be divisible by latent_patch_size={patch}."
            )
        token_h, token_w = height // patch, width // patch
        tokens = rearrange(latent, "C (H P) (W Q) -> (H W) (C P Q)", H=token_h, W=token_w, P=patch, Q=patch)
        return tokens, (token_h, token_w)

    def _patchify_latents_batched(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError("Latents must be (B,C,H,W) for batched patchify.")
        patch = self.latent_patch_size
        _, _, height, width = latents.shape
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"Latent spatial shape {(height, width)} must be divisible by latent_patch_size={patch}."
            )
        token_h, token_w = height // patch, width // patch
        return rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=token_h, W=token_w, P=patch, Q=patch)

    def _unpatchify_tokens(self, tokens: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        patch = self.latent_patch_size
        return rearrange(
            tokens,
            "B (H W) (C P Q) -> B C (H P) (W Q)",
            H=token_h,
            W=token_w,
            C=self.latent_channels,
            P=patch,
            Q=patch,
        )

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
                token_h, token_w = size
                ori_len = token_h * token_w
                local = unified_x[offset : offset + ori_len]
                out = rearrange(
                    local,
                    "(H W) (C P Q) -> C (H P) (W Q)",
                    H=token_h,
                    W=token_w,
                    C=self.latent_channels,
                    P=self.latent_patch_size,
                    Q=self.latent_patch_size,
                )
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
        token_valid_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        max_len = max(seq_lens)
        unified = pad_sequence(unified_list, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(freqs_list, batch_first=True, padding_value=0.0)

        key_mask = None
        has_batch_padding = any(seq_len < max_len for seq_len in seq_lens)
        has_internal_padding = False
        if token_valid_masks is not None:
            has_internal_padding = any((~valid_mask.bool()).any().item() for valid_mask in token_valid_masks)

        if has_batch_padding or has_internal_padding:
            key_mask = torch.zeros((len(seq_lens), 1, 1, max_len), device=device, dtype=dtype)
            for i, seq_len in enumerate(seq_lens):
                if token_valid_masks is not None:
                    valid_mask = token_valid_masks[i].to(device=device).bool()
                    invalid_index = torch.nonzero(~valid_mask, as_tuple=False).squeeze(-1)
                    if invalid_index.numel() > 0:
                        key_mask[i, 0, 0, invalid_index] = float("-inf")
                if seq_len < max_len:
                    key_mask[i, 0, 0, seq_len:] = float("-inf")

        return unified, unified_freqs, key_mask

    @staticmethod
    def _build_segmented_key_mask(
        original_lengths: List[int],
        padded_lengths: List[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if len(original_lengths) != len(padded_lengths):
            raise ValueError("original_lengths and padded_lengths must have same length.")

        if all(ori_len == pad_len for ori_len, pad_len in zip(original_lengths, padded_lengths)):
            return None

        total_len = sum(padded_lengths)
        key_mask = torch.zeros((1, 1, 1, total_len), device=device, dtype=dtype)
        offset = 0
        for ori_len, pad_len in zip(original_lengths, padded_lengths):
            if ori_len < pad_len:
                key_mask[:, :, :, offset + ori_len : offset + pad_len] = float("-inf")
            offset += pad_len

        return key_mask

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
    def _resolve_omni_condition_keep_flags(
        latent_items: List[Optional[torch.Tensor]],
        edit_latent_mask_item: Optional[List[bool]],
    ) -> List[bool]:
        if edit_latent_mask_item is None:
            return [True] * len(latent_items)

        flags = [bool(v) for v in edit_latent_mask_item]
        if len(flags) < len(latent_items):
            flags = flags + [True] * (len(latent_items) - len(flags))
        return flags[: len(latent_items)]

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

    def _reset_freq_cache_if_needed(self, device: torch.device):
        device_key = str(device)
        if self._freq_cache_device != device_key:
            self._freq_cache.clear()
            self._scaled_siglip_freq_cache.clear()
            self._freq_cache_device = device_key

    @staticmethod
    def _cache_lookup(cache: OrderedDict, key):
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    def _cache_store(self, cache: OrderedDict, key, value: torch.Tensor):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._max_freq_cache_entries:
            cache.popitem(last=False)

    def _build_2d_freqs(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        self._reset_freq_cache_if_needed(device)
        cache_key = (height, width)
        cached = self._cache_lookup(self._freq_cache, cache_key)
        if cached is not None:
            return cached
        img_freqs, _ = self.pos_embed([(1, height, width)], [1], device=device)
        cached = img_freqs[: height * width].contiguous()
        self._cache_store(self._freq_cache, cache_key, cached)
        return cached

    def _build_scaled_siglip_freqs(
        self,
        sig_h: int,
        sig_w: int,
        ref_h: int,
        ref_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        self._reset_freq_cache_if_needed(device)
        cache_key = (sig_h, sig_w, ref_h, ref_w)
        cached = self._cache_lookup(self._scaled_siglip_freq_cache, cache_key)
        if cached is not None:
            return cached
        ref_freqs = self._build_2d_freqs(ref_h, ref_w, device=device)
        ref_freqs = ref_freqs.view(ref_h, ref_w, -1)

        y_idx = torch.linspace(0, max(ref_h - 1, 0), steps=sig_h, device=device).round().long()
        x_idx = torch.linspace(0, max(ref_w - 1, 0), steps=sig_w, device=device).round().long()
        sampled = ref_freqs[y_idx][:, x_idx].reshape(sig_h * sig_w, -1).contiguous()
        self._cache_store(self._scaled_siglip_freq_cache, cache_key, sampled)
        return sampled

    def _build_omni_image_freqs(
        self,
        image_sizes: List[Optional[Tuple[int, int]]],
        image_lengths: List[int],
        txt_seq_len: int,
        device: torch.device,
        siglip_sizes: Optional[List[Optional[Tuple[int, int]]]] = None,
        siglip_lengths: Optional[List[int]] = None,
        siglip_ref_sizes: Optional[List[Optional[Tuple[int, int]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_img_freqs, _ = self.pos_embed([(1, 1, 1)], [1], device=device)
        template_freq = all_img_freqs[:1]
        segment_freqs = []

        for size, total_len in zip(image_sizes, image_lengths):
            if size is None:
                segment_freqs.append(template_freq.repeat(total_len, 1))
                continue

            height, width = size
            local = self._build_2d_freqs(height, width, device=device)
            if total_len > local.shape[0]:
                local = torch.cat([local, local[-1:].repeat(total_len - local.shape[0], 1)], dim=0)
            segment_freqs.append(local[:total_len])

        if siglip_sizes is not None and siglip_lengths is not None:
            if siglip_ref_sizes is None:
                siglip_ref_sizes = [None] * len(siglip_sizes)
            for size, ref_size, total_len in zip(siglip_sizes, siglip_ref_sizes, siglip_lengths):
                if size is None:
                    segment_freqs.append(template_freq.repeat(total_len, 1))
                    continue

                sig_h, sig_w = size
                if ref_size is not None:
                    ref_h, ref_w = ref_size
                    local = self._build_scaled_siglip_freqs(sig_h, sig_w, ref_h, ref_w, device=device)
                else:
                    local = self._build_2d_freqs(sig_h, sig_w, device=device)
                if total_len > local.shape[0]:
                    local = torch.cat([local, local[-1:].repeat(total_len - local.shape[0], 1)], dim=0)
                segment_freqs.append(local[:total_len])

        txt_ref_shapes = [(1, height, width) for size in image_sizes if size is not None for height, width in [size]]
        if siglip_sizes is not None:
            txt_ref_shapes.extend(
                (1, height, width) for size in siglip_sizes if size is not None for height, width in [size]
            )
        if len(txt_ref_shapes) == 0:
            txt_ref_shapes = [(1, 1, 1)]

        _, txt_freqs = self.pos_embed(txt_ref_shapes, [txt_seq_len], device=device)
        txt_freqs = txt_freqs[:txt_seq_len]

        img_freqs = torch.cat(segment_freqs, dim=0)
        return img_freqs, txt_freqs

    def _build_omni_unified_freqs(
        self,
        image_sizes: List[Optional[Tuple[int, int]]],
        image_lengths: List[int],
        txt_seq_len: int,
        device: torch.device,
        siglip_sizes: Optional[List[Optional[Tuple[int, int]]]] = None,
        siglip_lengths: Optional[List[int]] = None,
        siglip_ref_sizes: Optional[List[Optional[Tuple[int, int]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        image_freqs, _ = self._build_omni_image_freqs(
            image_sizes=image_sizes,
            image_lengths=image_lengths,
            txt_seq_len=1,
            device=device,
        )

        txt_ref_shapes = [(1, height, width) for size in image_sizes if size is not None for height, width in [size]]
        if siglip_sizes is not None:
            txt_ref_shapes.extend(
                (1, height, width) for size in siglip_sizes if size is not None for height, width in [size]
            )
        if len(txt_ref_shapes) == 0:
            txt_ref_shapes = [(1, 1, 1)]

        _, txt_freqs = self.pos_embed(txt_ref_shapes, [txt_seq_len], device=device)
        txt_freqs = txt_freqs[:txt_seq_len]

        siglip_freqs = None
        if siglip_sizes is not None and siglip_lengths is not None:
            if siglip_ref_sizes is None:
                siglip_ref_sizes = [None] * len(siglip_sizes)
            template_freq = self._build_2d_freqs(1, 1, device=device)[:1]
            siglip_segments = []
            for size, ref_size, total_len in zip(siglip_sizes, siglip_ref_sizes, siglip_lengths):
                if size is None:
                    siglip_segments.append(template_freq.repeat(total_len, 1))
                    continue

                sig_h, sig_w = size
                if ref_size is not None:
                    ref_h, ref_w = ref_size
                    local = self._build_scaled_siglip_freqs(sig_h, sig_w, ref_h, ref_w, device=device)
                else:
                    local = self._build_2d_freqs(sig_h, sig_w, device=device)
                if total_len > local.shape[0]:
                    local = torch.cat([local, local[-1:].repeat(total_len - local.shape[0], 1)], dim=0)
                siglip_segments.append(local[:total_len])

            siglip_freqs = torch.cat(siglip_segments, dim=0) if len(siglip_segments) > 0 else None

        return image_freqs, txt_freqs, siglip_freqs


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
        if latents.ndim == 4:
            latent_h, latent_w = latents.shape[-2:]
        else:
            latent_h, latent_w = height // self.latent_downsample_factor, width // self.latent_downsample_factor
        patched_masks = []
        for i in range(N):
            patched_mask = rearrange(
                entity_masks[i],
                "B C (H P) (W Q) -> B (H W) (C P Q)",
                H=latent_h,
                W=latent_w,
                P=2,
                Q=2,
            )
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
        edit_latent_mask: Optional[List[List[bool]]] = None,
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
            if edit_latent_mask is not None and (not isinstance(edit_latent_mask, list) or not isinstance(edit_latent_mask[0], list)):
                raise ValueError("edit_latent_mask must be List[List[bool]] when latents is omni-mode.")

            batch_size = len(latents)
            if prompt_emb.shape[0] != batch_size:
                raise ValueError(
                    f"Omni mode expects prompt_emb batch ({prompt_emb.shape[0]}) to match latent batch ({batch_size})."
                )
            if prompt_emb_mask is not None and prompt_emb_mask.shape[0] != batch_size:
                raise ValueError(
                    f"Omni mode expects prompt_emb_mask batch ({prompt_emb_mask.shape[0]}) to match latent batch ({batch_size})."
                )
            text_tokens = self.txt_in(self.txt_norm(prompt_emb))
            conditioning_noisy = self.time_text_embed(timestep, text_tokens.dtype)
            conditioning_clean = self.time_text_embed(torch.zeros_like(timestep), text_tokens.dtype)
            if self.text_pool_proj is not None:
                text_pool_cond = self._compute_text_pool_conditioning(prompt_emb, prompt_emb_mask)
                conditioning_noisy = conditioning_noisy + text_pool_cond
                conditioning_clean = conditioning_clean + text_pool_cond
            if conditioning_noisy.shape[0] == 1 and batch_size > 1:
                conditioning_noisy = conditioning_noisy.expand(batch_size, -1)
            elif conditioning_noisy.shape[0] != batch_size:
                raise ValueError(
                    f"Omni mode expects timestep embedding batch ({conditioning_noisy.shape[0]}) to be 1 or match latent batch ({batch_size})."
                )
            if conditioning_clean.shape[0] == 1 and batch_size > 1:
                conditioning_clean = conditioning_clean.expand(batch_size, -1)
            elif conditioning_clean.shape[0] != batch_size:
                raise ValueError(
                    f"Omni mode expects clean timestep embedding batch ({conditioning_clean.shape[0]}) to be 1 or match latent batch ({batch_size})."
                )

            unified_list = []
            freqs_list = []
            temb_list = []
            token_type_list = []
            valid_masks_list = []
            seq_lens = []
            x_sizes = []
            x_lengths = []
            x_pos_offsets = []

            prepared_image_tokens = []
            prepared_image_freqs_for_refiner = []
            prepared_image_temb = []
            prepared_image_token_types = []
            prepared_image_noise_masks = []
            prepared_image_valid_masks = []
            prepared_size_lists = []
            prepared_length_lists = []

            prepared_sig_tokens = []
            prepared_sig_freqs_for_refiner = []
            prepared_sig_shapes = []
            prepared_sig_lengths = []
            prepared_sig_noise_masks = []
            prepared_sig_valid_masks = []

            prepared_txt_tokens = []
            prepared_txt_freqs_for_refiner = []
            prepared_txt_seq_lens = []

            for b in range(batch_size):
                noise_flags = self._resolve_omni_image_noise_flags(
                    latents[b],
                    image_noise_mask[b] if image_noise_mask is not None and b < len(image_noise_mask) else None,
                )
                keep_flags = self._resolve_omni_condition_keep_flags(
                    latents[b],
                    edit_latent_mask[b] if edit_latent_mask is not None and b < len(edit_latent_mask) else None,
                )

                image_tokens_list = []
                size_list = []
                length_list = []
                image_token_noise_mask = []
                image_token_valid_mask = []
                for img_idx, img in enumerate(latents[b]):
                    local_noise_flag = noise_flags[img_idx]
                    local_keep_flag = keep_flags[img_idx]
                    if img is None:
                        pad_len = SEQ_MULTI_OF
                        image_tokens_list.append(
                            self.image_pad_token.to(device=prompt_emb.device, dtype=prompt_emb.dtype).repeat(pad_len, 1)
                        )
                        size_list.append(None)
                        length_list.append(pad_len)
                        image_token_noise_mask.extend([local_noise_flag] * pad_len)
                        image_token_valid_mask.extend([0] * pad_len)
                        continue
                    tokens, (h, w) = self._patchify_latent(img)
                    tokens, original_len, total_len = self._pad_tokens(tokens, pad_token=self.image_pad_token)
                    if local_keep_flag:
                        image_tokens_list.append(tokens)
                        image_token_valid_mask.extend([1] * original_len + [0] * (total_len - original_len))
                    else:
                        image_tokens_list.append(
                            self.image_pad_token.to(device=prompt_emb.device, dtype=prompt_emb.dtype).repeat(total_len, 1)
                        )
                        image_token_valid_mask.extend([0] * total_len)
                    size_list.append((h, w))
                    length_list.append(total_len)
                    image_token_noise_mask.extend([local_noise_flag] * total_len)

                image_tokens = torch.cat(image_tokens_list, dim=0)
                image_tokens = self.img_in(image_tokens)
                cond_token_mask = self._build_condition_token_mask(length_list, image_token_valid_mask).to(
                    device=image_tokens.device,
                    dtype=image_tokens.dtype,
                )
                edit_pool_cond = self._compute_edit_pool_conditioning(image_tokens, cond_token_mask)
                local_conditioning_noisy = conditioning_noisy[b : b + 1]
                local_conditioning_clean = conditioning_clean[b : b + 1]
                if edit_pool_cond is not None:
                    local_conditioning_noisy = local_conditioning_noisy + edit_pool_cond
                    local_conditioning_clean = local_conditioning_clean + edit_pool_cond
                    conditioning_noisy[b : b + 1] = local_conditioning_noisy
                    conditioning_clean[b : b + 1] = local_conditioning_clean
                image_temb = self._build_per_token_temb(
                    image_token_noise_mask,
                    local_conditioning_noisy,
                    local_conditioning_clean,
                )
                image_freqs_for_refiner, _ = self._build_omni_image_freqs(
                    image_sizes=size_list,
                    image_lengths=length_list,
                    txt_seq_len=1,
                    device=prompt_emb.device,
                )
                prepared_image_freqs_for_refiner.append(image_freqs_for_refiner)
                prepared_image_temb.append(image_temb.squeeze(0))
                image_token_types = []
                for image_idx, image_len in enumerate(length_list):
                    token_type = TOKEN_TYPE_TARGET_IMAGE if image_idx == len(length_list) - 1 else TOKEN_TYPE_COND_IMAGE
                    image_token_types.extend([token_type] * image_len)
                prepared_image_token_types.append(
                    torch.tensor(image_token_types, dtype=torch.long, device=prompt_emb.device)
                )

                txt_seq_len = (
                    int(prompt_emb_mask[b].sum().item())
                    if prompt_emb_mask is not None
                    else text_tokens.shape[1]
                )
                text_tokens_b = text_tokens[b, :txt_seq_len]

                _, txt_freqs_for_refiner = self._build_omni_image_freqs(
                    image_sizes=size_list,
                    image_lengths=length_list,
                    txt_seq_len=txt_seq_len,
                    device=prompt_emb.device,
                )
                txt_freqs_for_refiner = txt_freqs_for_refiner[:txt_seq_len]
                prepared_txt_tokens.append(text_tokens_b)
                prepared_txt_freqs_for_refiner.append(txt_freqs_for_refiner)
                prepared_txt_seq_lens.append(txt_seq_len)

                sig_tokens = None
                sig_shapes = []
                sig_lengths = []
                sig_token_noise_mask = []
                sig_token_valid_mask = []
                if siglip_feats is not None:
                    if self.siglip_embedder is None:
                        raise ValueError("siglip_feats provided but siglip_feat_dim is None.")
                    sig_raw_list = []
                    sig_raw_lens = []
                    for sig_idx, sig in enumerate(siglip_feats[b]):
                        local_noise_flag = noise_flags[sig_idx] if sig_idx < len(noise_flags) else noise_flags[-1]
                        if sig is None:
                            pad_len = SEQ_MULTI_OF
                            sig_shapes.append(None)
                            sig_lengths.append(pad_len)
                            sig_token_noise_mask.extend([local_noise_flag] * pad_len)
                            sig_token_valid_mask.extend([0] * pad_len)
                            continue
                        sig_tok, (sh, sw) = self._flatten_siglip(sig)
                        sig_shapes.append((sh, sw))
                        sig_original_len = sig_tok.shape[0]
                        sig_total_len = sig_original_len + ((-sig_original_len) % SEQ_MULTI_OF)
                        sig_lengths.append(sig_total_len)
                        sig_token_noise_mask.extend([local_noise_flag] * sig_total_len)
                        sig_raw_list.append(sig_tok)
                        sig_raw_lens.append(sig_tok.shape[0])
                        sig_token_valid_mask.extend([1] * sig_original_len + [0] * (sig_total_len - sig_original_len))

                    embedded_sig_chunks = []
                    if len(sig_raw_list) > 0:
                        sig_tokens_raw = torch.cat(sig_raw_list, dim=0)
                        sig_tokens_raw = self.siglip_embedder(sig_tokens_raw)
                        embedded_sig_chunks = list(sig_tokens_raw.split(sig_raw_lens, dim=0))

                    sig_list = []
                    sig_chunk_idx = 0
                    for sig_shape, sig_total_len in zip(sig_shapes, sig_lengths):
                        if sig_shape is None:
                            sig_list.append(
                                self.siglip_pad_token.to(device=prompt_emb.device, dtype=image_tokens.dtype).repeat(
                                    sig_total_len, 1
                                )
                            )
                            continue
                        sig_tok = embedded_sig_chunks[sig_chunk_idx]
                        sig_chunk_idx += 1
                        sig_tok, _, _ = self._pad_tokens(sig_tok, pad_token=self.siglip_pad_token)
                        sig_list.append(sig_tok)

                    sig_tokens = torch.cat(sig_list, dim=0)
                    sig_freqs_for_refiner, _ = self._build_omni_image_freqs(
                        image_sizes=[],
                        image_lengths=[],
                        txt_seq_len=1,
                        device=prompt_emb.device,
                        siglip_sizes=sig_shapes,
                        siglip_lengths=sig_lengths,
                        siglip_ref_sizes=size_list,
                    )
                    prepared_sig_freqs_for_refiner.append(sig_freqs_for_refiner)
                else:
                    prepared_sig_freqs_for_refiner.append(None)

                prepared_image_tokens.append(image_tokens)
                prepared_image_noise_masks.append(image_token_noise_mask)
                prepared_image_valid_masks.append(image_token_valid_mask)
                prepared_size_lists.append(size_list)
                prepared_length_lists.append(length_list)

                prepared_sig_tokens.append(sig_tokens)
                prepared_sig_shapes.append(sig_shapes)
                prepared_sig_lengths.append(sig_lengths)
                prepared_sig_noise_masks.append(sig_token_noise_mask)
                prepared_sig_valid_masks.append(sig_token_valid_mask)

            prepared_image_valid_mask_tensors = [
                torch.tensor(mask, dtype=torch.bool, device=prompt_emb.device)
                for mask in prepared_image_valid_masks
            ]
            batched_image_token_types = pad_sequence(
                prepared_image_token_types,
                batch_first=True,
                padding_value=TOKEN_TYPE_TEXT,
            )

            image_refiner_seq_lens = [sum(length_list) for length_list in prepared_length_lists]
            batched_image_tokens, batched_image_freqs_for_refiner, image_refiner_key_mask = self._build_padded_unified(
                prepared_image_tokens,
                prepared_image_freqs_for_refiner,
                image_refiner_seq_lens,
                dtype=text_tokens.dtype,
                device=prompt_emb.device,
                token_valid_masks=prepared_image_valid_mask_tensors,
            )
            batched_image_temb = pad_sequence(prepared_image_temb, batch_first=True, padding_value=0.0)
            for block in self.noise_refiner:
                batched_image_tokens = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                    hidden_states=batched_image_tokens,
                    temb=batched_image_temb,
                    token_type_ids=batched_image_token_types,
                    image_rotary_emb=batched_image_freqs_for_refiner,
                    attention_mask=image_refiner_key_mask,
                )

            for b in range(batch_size):
                image_len = image_refiner_seq_lens[b]
                prepared_image_tokens[b] = batched_image_tokens[b, :image_len, :]

            if siglip_feats is not None and self.siglip_refiner is not None:
                sig_refiner_seq_lens = [sum(lengths) for lengths in prepared_sig_lengths]
                prepared_sig_valid_mask_tensors = [
                    torch.tensor(mask, dtype=torch.bool, device=prompt_emb.device)
                    for mask in prepared_sig_valid_masks
                ]
                sig_freqs_for_batch = []
                for idx, freqs in enumerate(prepared_sig_freqs_for_refiner):
                    if freqs is None:
                        raise ValueError(f"Missing siglip refiner frequencies for sample {idx} in omni mode.")
                    sig_freqs_for_batch.append(freqs)
                batched_sig_tokens, batched_sig_freqs_for_refiner, sig_refiner_key_mask = self._build_padded_unified(
                    prepared_sig_tokens,
                    sig_freqs_for_batch,
                    sig_refiner_seq_lens,
                    dtype=text_tokens.dtype,
                    device=prompt_emb.device,
                    token_valid_masks=prepared_sig_valid_mask_tensors,
                )
                for block in self.siglip_refiner:
                    batched_sig_tokens = gradient_checkpoint_forward(
                        block,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                        hidden_states=batched_sig_tokens,
                        image_rotary_emb=batched_sig_freqs_for_refiner,
                        attention_mask=sig_refiner_key_mask,
                    )

                for b in range(batch_size):
                    sig_len = sig_refiner_seq_lens[b]
                    prepared_sig_tokens[b] = batched_sig_tokens[b, :sig_len, :]

            text_refiner_seq_lens = [int(v) for v in prepared_txt_seq_lens]
            batched_text_tokens, batched_text_freqs_for_refiner, text_refiner_key_mask = self._build_padded_unified(
                prepared_txt_tokens,
                prepared_txt_freqs_for_refiner,
                text_refiner_seq_lens,
                dtype=text_tokens.dtype,
                device=prompt_emb.device,
            )
            for block in self.context_refiner:
                batched_text_tokens = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                    hidden_states=batched_text_tokens,
                    temb=conditioning_noisy,
                    token_type_ids=torch.full(
                        (batched_text_tokens.shape[0], batched_text_tokens.shape[1]),
                        TOKEN_TYPE_TEXT,
                        dtype=torch.long,
                        device=prompt_emb.device,
                    ) if block.modulation else None,
                    image_rotary_emb=batched_text_freqs_for_refiner,
                    attention_mask=text_refiner_key_mask,
                )

            for b in range(batch_size):
                txt_seq_len = text_refiner_seq_lens[b]
                text_tokens_b = batched_text_tokens[b, :txt_seq_len, :]
                image_tokens = prepared_image_tokens[b]
                image_token_noise_mask = prepared_image_noise_masks[b]
                image_token_valid_mask = prepared_image_valid_masks[b]
                size_list = prepared_size_lists[b]
                length_list = prepared_length_lists[b]
                sig_tokens = prepared_sig_tokens[b]
                sig_shapes = prepared_sig_shapes[b]
                sig_lengths = prepared_sig_lengths[b]
                sig_token_noise_mask = prepared_sig_noise_masks[b]
                sig_token_valid_mask = prepared_sig_valid_masks[b]

                img_freqs, txt_freqs, sig_freqs = self._build_omni_unified_freqs(
                    image_sizes=size_list,
                    image_lengths=length_list,
                    txt_seq_len=txt_seq_len,
                    device=prompt_emb.device,
                    siglip_sizes=sig_shapes if sig_tokens is not None else None,
                    siglip_lengths=sig_lengths if sig_tokens is not None else None,
                    siglip_ref_sizes=size_list if sig_tokens is not None else None,
                )

                txt_token_noise_mask = [0] * txt_seq_len
                unified_token_noise_mask = image_token_noise_mask + txt_token_noise_mask + sig_token_noise_mask
                unified_token_valid_mask = image_token_valid_mask + [1] * txt_seq_len + sig_token_valid_mask
                unified_temb = self._build_per_token_temb(
                    unified_token_noise_mask,
                    conditioning_noisy[b : b + 1],
                    conditioning_clean[b : b + 1],
                ).squeeze(0)
                unified_token_types = []
                for image_idx, image_len in enumerate(length_list):
                    token_type = TOKEN_TYPE_TARGET_IMAGE if image_idx == len(length_list) - 1 else TOKEN_TYPE_COND_IMAGE
                    unified_token_types.extend([token_type] * image_len)
                unified_token_types.extend([TOKEN_TYPE_TEXT] * txt_seq_len)
                if sig_tokens is not None:
                    unified_token_types = unified_token_types + [TOKEN_TYPE_SIGLIP] * sig_tokens.shape[0]

                # Align omni unified order with base branch: [x, cap, siglip]
                unified = torch.cat([image_tokens, text_tokens_b] + ([sig_tokens] if sig_tokens is not None else []), dim=0)
                unified_freqs = torch.cat(
                    [img_freqs, txt_freqs] + ([sig_freqs] if sig_freqs is not None else []),
                    dim=0,
                )

                unified_list.append(unified)
                freqs_list.append(unified_freqs)
                temb_list.append(unified_temb)
                token_type_list.append(torch.tensor(unified_token_types, dtype=torch.long, device=prompt_emb.device))
                valid_masks_list.append(torch.tensor(unified_token_valid_mask, dtype=torch.bool, device=prompt_emb.device))
                seq_lens.append(unified.shape[0])
                x_sizes.append(size_list)
                x_lengths.append(length_list)
                x_pos_offsets.append((0, sum(length_list)))

            unified, unified_freqs, key_mask = self._build_padded_unified(
                unified_list,
                freqs_list,
                seq_lens,
                dtype=text_tokens.dtype,
                device=prompt_emb.device,
                token_valid_masks=valid_masks_list,
            )
            unified_temb = pad_sequence(temb_list, batch_first=True, padding_value=0.0)
            unified_token_types = pad_sequence(token_type_list, batch_first=True, padding_value=TOKEN_TYPE_TEXT)
            if self.token_type_embed is not None:
                unified = unified + self.token_type_embed(self._resolve_token_type_embed_ids(unified_token_types))

            use_tread_routing = self.training and self.tread_router is not None
            route_idx = 0
            current_route = self.tread_routes[route_idx] if use_tread_routing else None
            route_ids_keep = None
            routed_key_mask = None
            routed_unified_temb = unified_temb
            routed_unified_token_types = unified_token_types
            route_original_unified = None
            route_original_freqs = None
            route_original_temb = None
            route_original_token_types = None

            for block_idx, block in enumerate(self.transformer_blocks):
                if use_tread_routing and current_route is not None and block_idx == current_route["start_layer_idx"]:
                    route_original_unified = unified
                    route_original_freqs = unified_freqs
                    route_original_temb = unified_temb
                    route_original_token_types = unified_token_types
                    route_ids_keep = self.tread_router.get_mask(unified, selection_rate=current_route["selection_ratio"])
                    unified = self.tread_router.start_route(unified, route_ids_keep)
                    unified_freqs = self.tread_router.start_route(unified_freqs, route_ids_keep)
                    routed_unified_temb = self.tread_router.start_route(unified_temb, route_ids_keep)
                    routed_unified_token_types = self.tread_router.start_route(
                        unified_token_types.unsqueeze(-1), route_ids_keep
                    ).squeeze(-1)
                    if key_mask is not None:
                        routed_key_mask = key_mask.gather(
                            -1,
                            route_ids_keep.unsqueeze(1).unsqueeze(1).expand(-1, key_mask.shape[1], key_mask.shape[2], -1),
                        )
                    else:
                        routed_key_mask = None

                active_key_mask = routed_key_mask if route_ids_keep is not None else key_mask
                active_temb = routed_unified_temb if route_ids_keep is not None else unified_temb
                active_token_types = None
                if self.use_omni_token_type_modulation:
                    active_token_types = routed_unified_token_types if route_ids_keep is not None else unified_token_types

                unified = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                    hidden_states=unified,
                    temb=active_temb,
                    token_type_ids=active_token_types,
                    image_rotary_emb=unified_freqs,
                    attention_mask=active_key_mask,
                )

                if use_tread_routing and current_route is not None and block_idx == current_route["end_layer_idx"]:
                    unified = self.tread_router.end_route(unified, route_ids_keep, route_original_unified)
                    unified_freqs = self.tread_router.end_route(unified_freqs, route_ids_keep, route_original_freqs)
                    unified_temb = self.tread_router.end_route(routed_unified_temb, route_ids_keep, route_original_temb)
                    unified_token_types = self.tread_router.end_route(
                        routed_unified_token_types.unsqueeze(-1),
                        route_ids_keep,
                        route_original_token_types.unsqueeze(-1),
                    ).squeeze(-1)
                    route_ids_keep = None
                    routed_key_mask = None
                    routed_unified_temb = unified_temb
                    routed_unified_token_types = unified_token_types
                    route_original_unified = None
                    route_original_freqs = None
                    route_original_temb = None
                    route_original_token_types = None
                    route_idx += 1
                    current_route = self.tread_routes[route_idx] if route_idx < len(self.tread_routes) else None

            # Extract only image tokens before norm_out/proj_out to avoid
            # wasting computation on text/siglip tokens that will be discarded.
            image_token_list = []
            for i, (start, end) in enumerate(x_pos_offsets):
                img_tok = unified[i, start:end, :].unsqueeze(0)
                img_tok = self.norm_out(img_tok, conditioning_noisy[i:i+1])
                img_tok = self.proj_out(img_tok)
                image_token_list.append(img_tok.squeeze(0))

            # _unpatchify_omni expects per-sample tokens; since we already
            # sliced to [start:end], use zero-based offsets.
            zero_offsets = [(0, end - start) for start, end in x_pos_offsets]
            outputs = self._unpatchify_omni(
                image_token_list,
                x_sizes, x_lengths, zero_offsets,
            )
            outputs = [out.unsqueeze(0) for out in outputs]
            return torch.cat(outputs, dim=0)

        if latents.ndim == 4:
            latent_h_full, latent_w_full = latents.shape[-2:]
            image_tokens = self._patchify_latents_batched(latents)
            latent_h, latent_w = latent_h_full // self.latent_patch_size, latent_w_full // self.latent_patch_size
            return_latents_4d = True
        else:
            image_tokens = latents
            return_latents_4d = False
            if height is not None and width is not None:
                latent_h_full = height // self.latent_downsample_factor
                latent_w_full = width // self.latent_downsample_factor
                if latent_h_full % self.latent_patch_size != 0 or latent_w_full % self.latent_patch_size != 0:
                    raise ValueError(
                        f"Latent spatial shape {(latent_h_full, latent_w_full)} inferred from height/width must be divisible "
                        f"by latent_patch_size={self.latent_patch_size}."
                    )
                latent_h, latent_w = latent_h_full // self.latent_patch_size, latent_w_full // self.latent_patch_size
            else:
                seq_len = image_tokens.shape[1]
                side = int(math.sqrt(seq_len))
                if side * side != seq_len:
                    raise ValueError("height/width required when latents are tokenized and sequence is not square.")
                latent_h = latent_w = side

        expected_channels = int(self.img_in.in_features)
        actual_channels = int(image_tokens.shape[-1])
        if actual_channels != expected_channels:
            raise ValueError(
                f"Latent token dim mismatch for ComplextroImageDiT: got {actual_channels}, expected {expected_channels}. "
                "Please align the incoming latent representation with DiT latent_channels/latent_patch_size."
            )

        text_tokens = self.txt_in(self.txt_norm(prompt_emb))
        image_tokens = self.img_in(image_tokens)

        txt_seq_lens = (
            [int(v) for v in prompt_emb_mask.sum(dim=1).tolist()]
            if prompt_emb_mask is not None
            else [text_tokens.shape[1]] * text_tokens.shape[0]
        )
        max_txt_len = max(txt_seq_lens)
        text_tokens = text_tokens[:, :max_txt_len]

        text_key_mask = None
        if prompt_emb_mask is not None:
            if any(txt_len < max_txt_len for txt_len in txt_seq_lens):
                text_key_mask = torch.zeros(
                    (len(txt_seq_lens), 1, 1, max_txt_len),
                    device=text_tokens.device,
                    dtype=text_tokens.dtype,
                )
                for i, txt_len in enumerate(txt_seq_lens):
                    if txt_len < max_txt_len:
                        text_key_mask[i, 0, 0, txt_len:] = float("-inf")

        text_freqs_list = []
        for txt_len in txt_seq_lens:
            _, txt_freqs = self.pos_embed([(1, latent_h, latent_w)], [int(txt_len)], device=image_tokens.device)
            text_freqs_list.append(txt_freqs[: int(txt_len)])
        text_freqs_for_refiner = pad_sequence(text_freqs_list, batch_first=True, padding_value=0.0)

        x_seqlen = image_tokens.shape[1]
        image_freqs_single = self._build_2d_freqs(latent_h, latent_w, device=image_tokens.device)
        if x_seqlen > image_freqs_single.shape[0]:
            image_freqs_single = torch.cat(
                [image_freqs_single, image_freqs_single[-1:].repeat(x_seqlen - image_freqs_single.shape[0], 1)],
                dim=0,
            )
        image_freqs_for_refiner = image_freqs_single[:x_seqlen].unsqueeze(0).repeat(image_tokens.shape[0], 1, 1)

        conditioning = self.time_text_embed(timestep, image_tokens.dtype)
        if self.text_pool_proj is not None:
            conditioning = conditioning + self._compute_text_pool_conditioning(prompt_emb, prompt_emb_mask)
        image_token_types = torch.full(
            (image_tokens.shape[0], image_tokens.shape[1]),
            TOKEN_TYPE_TARGET_IMAGE,
            dtype=torch.long,
            device=image_tokens.device,
        )

        # Refine image tokens (noise) and text tokens (context)
        for block in self.noise_refiner:
            image_tokens = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=image_tokens,
                temb=conditioning,
                token_type_ids=image_token_types,
                image_rotary_emb=image_freqs_for_refiner,
            )

        for block in self.context_refiner:
            text_tokens = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=text_tokens,
                temb=conditioning,
                token_type_ids=torch.full(
                    (text_tokens.shape[0], text_tokens.shape[1]),
                    TOKEN_TYPE_TEXT,
                    dtype=torch.long,
                    device=text_tokens.device,
                ) if block.modulation else None,
                image_rotary_emb=text_freqs_for_refiner,
                attention_mask=text_key_mask,
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
            siglip_freqs_for_refiner = self._build_2d_freqs(siglip_shape[1], siglip_shape[2], device=image_tokens.device)
            siglip_freqs_for_refiner = siglip_freqs_for_refiner[:siglip_tokens.shape[1]]
            siglip_freqs_for_refiner = siglip_freqs_for_refiner.unsqueeze(0).repeat(siglip_tokens.shape[0], 1, 1)
            for block in self.siglip_refiner:
                siglip_tokens = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                    hidden_states=siglip_tokens,
                    image_rotary_emb=siglip_freqs_for_refiner,
                )

        # RoPE for unified sequence (text + image + optional siglip)
        # Align basic unified order with z_image_dit: [x, cap]
        # (keep optional siglip support by appending after cap when present)
        unified_list = []
        freqs_list = []
        token_type_list = []
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
                unified_type_b = [TOKEN_TYPE_TARGET_IMAGE] * x_seqlen + [TOKEN_TYPE_TEXT] * txt_len + [TOKEN_TYPE_SIGLIP] * siglip_len
            else:
                unified_b = torch.cat([image_b, text_b], dim=0)
                freqs_b = torch.cat([img_freqs, txt_freqs], dim=0)
                unified_type_b = [TOKEN_TYPE_TARGET_IMAGE] * x_seqlen + [TOKEN_TYPE_TEXT] * txt_len

            unified_list.append(unified_b)
            freqs_list.append(freqs_b)
            token_type_list.append(torch.tensor(unified_type_b, dtype=torch.long, device=image_tokens.device))
            seq_lens.append(unified_b.shape[0])
            x_pos_offsets.append((0, x_seqlen))

        unified, unified_freqs, key_mask = self._build_padded_unified(
            unified_list,
            freqs_list,
            seq_lens,
            dtype=image_tokens.dtype,
            device=image_tokens.device,
        )
        unified_token_types = pad_sequence(token_type_list, batch_first=True, padding_value=TOKEN_TYPE_TEXT)
        if self.token_type_embed is not None:
            unified = unified + self.token_type_embed(self._resolve_token_type_embed_ids(unified_token_types))

        use_tread_routing = self.training and self.tread_router is not None
        route_idx = 0
        current_route = self.tread_routes[route_idx] if use_tread_routing else None
        route_ids_keep = None
        routed_key_mask = None
        route_original_unified = None
        route_original_freqs = None
        route_original_token_types = None
        routed_unified_token_types = unified_token_types

        for block_idx, block in enumerate(self.transformer_blocks):
            if use_tread_routing and current_route is not None and block_idx == current_route["start_layer_idx"]:
                route_original_unified = unified
                route_original_freqs = unified_freqs
                route_original_token_types = unified_token_types
                route_ids_keep = self.tread_router.get_mask(unified, selection_rate=current_route["selection_ratio"])
                unified = self.tread_router.start_route(unified, route_ids_keep)
                unified_freqs = self.tread_router.start_route(unified_freqs, route_ids_keep)
                routed_unified_token_types = self.tread_router.start_route(
                    unified_token_types.unsqueeze(-1), route_ids_keep
                ).squeeze(-1)
                if key_mask is not None:
                    routed_key_mask = key_mask.gather(
                        -1,
                        route_ids_keep.unsqueeze(1).unsqueeze(1).expand(-1, key_mask.shape[1], key_mask.shape[2], -1),
                    )
                else:
                    routed_key_mask = None

            active_key_mask = routed_key_mask if route_ids_keep is not None else key_mask
            active_token_types = None
            if self.use_unified_token_type_modulation:
                active_token_types = routed_unified_token_types if route_ids_keep is not None else unified_token_types

            unified = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=unified,
                temb=conditioning,
                token_type_ids=active_token_types,
                image_rotary_emb=unified_freqs,
                attention_mask=active_key_mask,
            )

            if use_tread_routing and current_route is not None and block_idx == current_route["end_layer_idx"]:
                unified = self.tread_router.end_route(unified, route_ids_keep, route_original_unified)
                unified_freqs = self.tread_router.end_route(unified_freqs, route_ids_keep, route_original_freqs)
                unified_token_types = self.tread_router.end_route(
                    routed_unified_token_types.unsqueeze(-1),
                    route_ids_keep,
                    route_original_token_types.unsqueeze(-1),
                ).squeeze(-1)
                route_ids_keep = None
                routed_key_mask = None
                routed_unified_token_types = unified_token_types
                route_original_unified = None
                route_original_freqs = None
                route_original_token_types = None
                route_idx += 1
                current_route = self.tread_routes[route_idx] if route_idx < len(self.tread_routes) else None

        image_tokens = []
        for b, (start, end) in enumerate(x_pos_offsets):
            image_tokens.append(unified[b, start:end, :])
        image_tokens = torch.stack(image_tokens, dim=0)
        image_tokens = self.norm_out(image_tokens, conditioning)
        image_tokens = self.proj_out(image_tokens)

        if return_latents_4d:
            latents = self._unpatchify_tokens(image_tokens, latent_h, latent_w)
            return latents

        return image_tokens
