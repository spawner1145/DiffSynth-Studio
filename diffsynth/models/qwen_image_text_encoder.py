import torch
from typing import Optional, Union


class QwenImageTextEncoder(torch.nn.Module):
    def __init__(self, model_type="qwen2_5_vl", model_size="7B"):
        super().__init__()
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel

        model_dict = {
            "qwen2_5_vl": Qwen2_5_VLModel,
        }

        config_dict = {
            "qwen2_5_vl": {
                "7B": Qwen2_5_VLConfig(**{
                    "architectures": [
                        "Qwen2_5_VLForConditionalGeneration"
                    ],
                    "attention_dropout": 0.0,
                    "bos_token_id": 151643,
                    "eos_token_id": 151645,
                    "hidden_act": "silu",
                    "hidden_size": 3584,
                    "image_token_id": 151655,
                    "initializer_range": 0.02,
                    "intermediate_size": 18944,
                    "max_position_embeddings": 128000,
                    "max_window_layers": 28,
                    "model_type": "qwen2_5_vl",
                    "num_attention_heads": 28,
                    "num_hidden_layers": 28,
                    "num_key_value_heads": 4,
                    "rms_norm_eps": 1e-06,
                    "rope_scaling": {
                        "mrope_section": [
                            16,
                            24,
                            24
                        ],
                        "rope_type": "default",
                        "type": "default"
                    },
                    "rope_theta": 1000000.0,
                    "sliding_window": 32768,
                    "text_config": {
                        "architectures": [
                            "Qwen2_5_VLForConditionalGeneration"
                        ],
                        "attention_dropout": 0.0,
                        "bos_token_id": 151643,
                        "eos_token_id": 151645,
                        "hidden_act": "silu",
                        "hidden_size": 3584,
                        "image_token_id": None,
                        "initializer_range": 0.02,
                        "intermediate_size": 18944,
                        "layer_types": [
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention",
                        "full_attention"
                        ],
                        "max_position_embeddings": 128000,
                        "max_window_layers": 28,
                        "model_type": "qwen2_5_vl_text",
                        "num_attention_heads": 28,
                        "num_hidden_layers": 28,
                        "num_key_value_heads": 4,
                        "rms_norm_eps": 1e-06,
                        "rope_scaling": {
                        "mrope_section": [
                            16,
                            24,
                            24
                        ],
                        "rope_type": "default",
                        "type": "default"
                        },
                        "rope_theta": 1000000.0,
                        "sliding_window": None,
                        "torch_dtype": "float32",
                        "use_cache": True,
                        "use_sliding_window": False,
                        "video_token_id": None,
                        "vision_end_token_id": 151653,
                        "vision_start_token_id": 151652,
                        "vision_token_id": 151654,
                        "vocab_size": 152064
                    },
                    "tie_word_embeddings": False,
                    "torch_dtype": "float32",
                    "transformers_version": "4.54.0",
                    "use_cache": True,
                    "use_sliding_window": False,
                    "video_token_id": 151656,
                    "vision_config": {
                        "depth": 32,
                        "fullatt_block_indexes": [
                            7,
                            15,
                            23,
                            31
                        ],
                        "hidden_act": "silu",
                        "hidden_size": 1280,
                        "in_channels": 3,
                        "in_chans": 3,
                        "initializer_range": 0.02,
                        "intermediate_size": 3420,
                        "model_type": "qwen2_5_vl",
                        "num_heads": 16,
                        "out_hidden_size": 3584,
                        "patch_size": 14,
                        "spatial_merge_size": 2,
                        "spatial_patch_size": 14,
                        "temporal_patch_size": 2,
                        "tokens_per_second": 2,
                        "torch_dtype": "float32",
                        "window_size": 112
                    },
                    "vision_end_token_id": 151653,
                    "vision_start_token_id": 151652,
                    "vision_token_id": 151654,
                    "vocab_size": 152064
                }),
            },
        }

        if model_type in ("qwen3_5", "qwen3_5_moe"):
            from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration

            model_dict.update({
                "qwen3_5": Qwen3_5ForConditionalGeneration,
                "qwen3_5_moe": Qwen3_5MoeForConditionalGeneration,
            })
            config_dict["qwen3_5"] = {
                "0.8B": Qwen3_5Config(**{
                    "architectures": [
                        "Qwen3_5ForConditionalGeneration"
                    ],
                    "image_token_id": 248056,
                    "model_type": "qwen3_5",
                    "text_config": {
                        "attention_bias": False,
                        "attention_dropout": 0.0,
                        "attn_output_gate": True,
                        "dtype": "bfloat16",
                        "eos_token_id": 248044,
                        "full_attention_interval": 4,
                        "head_dim": 256,
                        "hidden_act": "silu",
                        "hidden_size": 1024,
                        "initializer_range": 0.02,
                        "intermediate_size": 3584,
                        "layer_types": [
                            "linear_attention",
                            "linear_attention",
                            "linear_attention",
                            "full_attention",
                            "linear_attention",
                            "linear_attention",
                            "linear_attention",
                            "full_attention",
                            "linear_attention",
                            "linear_attention",
                            "linear_attention",
                            "full_attention",
                            "linear_attention",
                            "linear_attention",
                            "linear_attention",
                            "full_attention",
                            "linear_attention",
                            "linear_attention",
                            "linear_attention",
                            "full_attention",
                            "linear_attention",
                            "linear_attention",
                            "linear_attention",
                            "full_attention"
                        ],
                        "linear_conv_kernel_dim": 4,
                        "linear_key_head_dim": 128,
                        "linear_num_key_heads": 16,
                        "linear_num_value_heads": 16,
                        "linear_value_head_dim": 128,
                        "max_position_embeddings": 262144,
                        "mlp_only_layers": [],
                        "model_type": "qwen3_5_text",
                        "mtp_num_hidden_layers": 1,
                        "mtp_use_dedicated_embeddings": False,
                        "num_attention_heads": 8,
                        "num_hidden_layers": 24,
                        "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-06,
                        "tie_word_embeddings": True,
                        "use_cache": True,
                        "vocab_size": 248320,
                        "mamba_ssm_dtype": "float32",
                        "rope_parameters": {
                            "mrope_interleaved": True,
                            "mrope_section": [
                                11,
                                11,
                                10
                            ],
                            "rope_type": "default",
                            "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25
                        }
                    },
                    "tie_word_embeddings": True,
                    "transformers_version": "4.57.0.dev0",
                    "video_token_id": 248057,
                    "vision_config": {
                        "deepstack_visual_indexes": [],
                        "depth": 12,
                        "hidden_act": "gelu_pytorch_tanh",
                        "hidden_size": 768,
                        "in_channels": 3,
                        "initializer_range": 0.02,
                        "intermediate_size": 3072,
                        "model_type": "qwen3_5",
                        "num_heads": 12,
                        "num_position_embeddings": 2304,
                        "out_hidden_size": 1024,
                        "patch_size": 16,
                        "spatial_merge_size": 2,
                        "temporal_patch_size": 2
                    },
                    "vision_end_token_id": 248054,
                    "vision_start_token_id": 248053
                }),
            }

        if model_type not in model_dict:
            raise ValueError(f"Unsupported model_type: {model_type}")
        if model_type not in config_dict or model_size not in config_dict[model_type]:
            raise ValueError(f"Unsupported model config: model_type={model_type}, model_size={model_size}")

        config = config_dict[model_type][model_size]
        self.model_type = model_type
        self.model = model_dict[model_type](config)
        if model_type == "qwen2_5_vl":
            self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.config = config
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        output_attentions = False
        output_hidden_states = True

        if self.model_type == "qwen2_5_vl":
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        return outputs.hidden_states
