import torch


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def _has_te_linear(model):
    """Check if model contains any Transformer Engine Linear modules."""
    try:
        import transformer_engine.pytorch as te
        return any(isinstance(m, te.Linear) for m in model.modules())
    except ImportError:
        return False


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    **kwargs,
):
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            model_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        # Use TE checkpoint when model contains TE Linear modules
        if _has_te_linear(model):
            import transformer_engine.pytorch as te
            model_output = te.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
        else:
            model_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    else:
        model_output = model(*args, **kwargs)
    return model_output
