def QwenImageTextEncoderStateDictConverter(state_dict):
    keys = list(state_dict.keys())

    has_wrapped_qwen35_prefix = any(k.startswith("model.model.visual.") for k in keys) or any(
        k.startswith("model.model.language_model.") for k in keys
    )

    is_qwen35_conditional_checkpoint = (
        any(k.startswith("model.visual.") for k in keys)
        and any(k.startswith("model.language_model.") for k in keys)
        and not has_wrapped_qwen35_prefix
    )

    state_dict_ = {}
    for k, v in state_dict.items():
        if k.startswith("mtp."):
            continue

        if is_qwen35_conditional_checkpoint:
            if k.startswith("model."):
                k = "model." + k
            elif k.startswith("lm_head."):
                k = "model." + k
            state_dict_[k] = v
            continue

        if k.startswith("visual."):
            k = "model." + k
        elif k.startswith("model.") and not k.startswith("model.visual.") and not k.startswith("model.language_model."):
            k = k.replace("model.", "model.language_model.", 1)
        state_dict_[k] = v

    if "model.lm_head.weight" not in state_dict_:
        if "model.model.language_model.embed_tokens.weight" in state_dict_:
            state_dict_["model.lm_head.weight"] = state_dict_["model.model.language_model.embed_tokens.weight"]
        elif "model.language_model.embed_tokens.weight" in state_dict_:
            state_dict_["model.lm_head.weight"] = state_dict_["model.language_model.embed_tokens.weight"]

    return state_dict_
