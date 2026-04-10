import torch

from diffsynth.models.complextro_dit import ComplextroImageDiT


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    # 按你训练里用的配置来构造模型
    complextro_model_config = {
        "num_layers": 10,
        "num_refiner_layers": 0,
        "text_embed_dim": 2048, # 额外要加的文本模态输入维度
        "hidden_size": 1024,
        "num_attention_heads": 32,
        "attention_head_dim": 32,
        "rope_axes_dim": [8, 12, 12],
        "enable_tread_routing": False,
        "tread_routes": [
            {
                "selection_ratio": 0.5,
                "start_layer_idx": 0,
                "end_layer_idx": 1,
            }
        ],
        "use_text_modulation": True,
        "shared_modulation_group_size": "all",
    }

    model = ComplextroImageDiT(**complextro_model_config)

    total, trainable = count_parameters(model)
    print(f"Total params:      {total:,}")
    print(f"Trainable params:  {trainable:,}")
    print(f"Total (M):         {total / 1e6:.3f} M")
    print(f"Trainable (M):     {trainable / 1e6:.3f} M")


if __name__ == "__main__":
    main()