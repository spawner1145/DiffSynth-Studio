def QwenImageTextEncoderStateDictConverter(state_dict):
    has_new_prefix = False
    for k in state_dict:
        if k.startswith("model.visual.") or k.startswith("model.language_model."):
            has_new_prefix = True
            break

    state_dict_ = {}
    for k in state_dict:
        v = state_dict[k]
        if has_new_prefix:
            state_dict_[k] = v
            continue

        if k.startswith("visual."):
            k = "model." + k
        elif k.startswith("model."):
            k = k.replace("model.", "model.language_model.", 1)
        state_dict_[k] = v
    return state_dict_
