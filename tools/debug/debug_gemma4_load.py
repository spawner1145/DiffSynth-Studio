import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path


def _read_state_dict(path, torch_dtype=None, device="cpu"):
    from diffsynth.core.loader.file import load_state_dict

    return load_state_dict(path, torch_dtype=torch_dtype, device=device)


def _import_symbol(qualified_name: str):
    import importlib

    module_name, symbol_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _safe_list(iterable, limit=None):
    items = list(iterable)
    if limit is not None:
        return items[:limit]
    return items


def main():
    parser = argparse.ArgumentParser(description="Debug Gemma4 state_dict loading and dump full errors to txt.")
    parser.add_argument("--model-path", required=True, help="Path to model file (.safetensors/.bin/.pth)")
    parser.add_argument("--output", default=None, help="Output txt path. Default: tools/debug/gemma4_load_report_<timestamp>.txt")
    parser.add_argument("--model-type", default="gemma4")
    parser.add_argument("--model-size", default="2B")
    parser.add_argument("--use-converter", action="store_true", help="Apply QwenImageTextEncoderStateDictConverter before loading")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="none", choices=["none", "float16", "bfloat16", "float32"])
    parser.add_argument("--preview-limit", type=int, default=200000, help="How many keys to preview in each section")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or f"tools/debug/gemma4_load_report_{timestamp}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder

    dtype_map = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    report = []

    def log(title, content=""):
        report.append(f"\n===== {title} =====\n")
        if isinstance(content, str):
            report.append(content)
        else:
            report.append(json.dumps(content, ensure_ascii=False, indent=2))
        report.append("\n")

    try:
        log("ARGS", vars(args))

        state_dict = _read_state_dict(args.model_path, torch_dtype=torch_dtype, device=args.device)
        original_keys = list(state_dict.keys())
        log("ORIGINAL_KEY_COUNT", str(len(original_keys)))
        log("ORIGINAL_KEY_PREVIEW", "\n".join(original_keys[: args.preview_limit]))

        converted_state_dict = state_dict
        if args.use_converter:
            converter = _import_symbol(
                "diffsynth.utils.state_dict_converters.qwen_image_text_encoder.QwenImageTextEncoderStateDictConverter"
            )
            converted_state_dict = converter(state_dict)

        converted_keys = list(converted_state_dict.keys())
        log("CONVERTED_KEY_COUNT", str(len(converted_keys)))
        log("CONVERTED_KEY_PREVIEW", "\n".join(converted_keys[: args.preview_limit]))

        model = QwenImageTextEncoder(model_type=args.model_type, model_size=args.model_size)
        model_keys = list(model.state_dict().keys())
        log("MODEL_KEY_COUNT", str(len(model_keys)))
        log("MODEL_KEY_PREVIEW", "\n".join(model_keys[: args.preview_limit]))

        converted_key_set = set(converted_keys)
        model_key_set = set(model_keys)

        unexpected = sorted(converted_key_set - model_key_set)
        missing = sorted(model_key_set - converted_key_set)

        log("UNEXPECTED_KEY_COUNT", str(len(unexpected)))
        log("UNEXPECTED_KEY_PREVIEW", "\n".join(_safe_list(unexpected, args.preview_limit)))
        log("MISSING_KEY_COUNT", str(len(missing)))
        log("MISSING_KEY_PREVIEW", "\n".join(_safe_list(missing, args.preview_limit)))

        shape_mismatch = []
        for k in sorted(converted_key_set & model_key_set):
            try:
                src_shape = tuple(converted_state_dict[k].shape)
                dst_shape = tuple(model.state_dict()[k].shape)
                if src_shape != dst_shape:
                    shape_mismatch.append({
                        "key": k,
                        "checkpoint_shape": src_shape,
                        "model_shape": dst_shape,
                    })
            except Exception:
                pass
        log("SHAPE_MISMATCH_COUNT", str(len(shape_mismatch)))
        log("SHAPE_MISMATCH_PREVIEW", shape_mismatch[: args.preview_limit])

        try:
            model.load_state_dict(converted_state_dict, assign=True)
            log("LOAD_RESULT", "load_state_dict succeeded")
        except Exception as e:
            log("LOAD_EXCEPTION_TYPE", type(e).__name__)
            log("LOAD_EXCEPTION_MESSAGE", str(e))
            log("LOAD_TRACEBACK", traceback.format_exc())

    except Exception as e:
        log("FATAL_EXCEPTION_TYPE", type(e).__name__)
        log("FATAL_EXCEPTION_MESSAGE", str(e))
        log("FATAL_TRACEBACK", traceback.format_exc())

    output_path.write_text("".join(report), encoding="utf-8")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
