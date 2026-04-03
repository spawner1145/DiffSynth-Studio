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

    is_gemma4_nested_multimodal_checkpoint = any(k.startswith("model.language_model.audio_tower.") for k in keys) or any(
        k.startswith("model.language_model.vision_tower.") for k in keys
    )

    state_dict_ = {}
    for k, v in state_dict.items():
        if k.startswith("mtp."):
            continue

        if is_gemma4_nested_multimodal_checkpoint:
            if k.startswith("model.language_model.audio_tower."):
                k = k.replace("model.language_model.audio_tower.", "model.audio_tower.", 1)
            elif k.startswith("model.language_model.vision_tower."):
                k = k.replace("model.language_model.vision_tower.", "model.vision_tower.", 1)
            elif k.startswith("model.language_model.embed_audio."):
                k = k.replace("model.language_model.embed_audio.", "model.embed_audio.", 1)
            elif k.startswith("model.language_model.embed_vision."):
                k = k.replace("model.language_model.embed_vision.", "model.embed_vision.", 1)
            elif k.startswith("model.language_model.lm_head."):
                k = k.replace("model.language_model.lm_head.", "lm_head.", 1)
            state_dict_[k] = v
            continue

        if has_wrapped_qwen35_prefix:
            if k.startswith("lm_head."):
                k = "model." + k
            state_dict_[k] = v
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
        elif (
            k.startswith("model.")
            and not k.startswith("model.visual.")
            and not k.startswith("model.language_model.")
            and not k.startswith("model.model.")
        ):
            k = k.replace("model.", "model.language_model.", 1)
        state_dict_[k] = v

    is_qwen35_like = is_qwen35_conditional_checkpoint or has_wrapped_qwen35_prefix

    if is_qwen35_like and "model.lm_head.weight" not in state_dict_:
        if "model.model.language_model.embed_tokens.weight" in state_dict_:
            state_dict_["model.lm_head.weight"] = state_dict_["model.model.language_model.embed_tokens.weight"]
        elif "model.language_model.embed_tokens.weight" in state_dict_:
            state_dict_["model.lm_head.weight"] = state_dict_["model.language_model.embed_tokens.weight"]

    if is_gemma4_nested_multimodal_checkpoint and "lm_head.weight" not in state_dict_:
        if "model.language_model.embed_tokens.weight" in state_dict_:
            state_dict_["lm_head.weight"] = state_dict_["model.language_model.embed_tokens.weight"]

    return state_dict_
