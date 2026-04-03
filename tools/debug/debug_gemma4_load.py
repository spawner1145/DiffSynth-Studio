import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Debug Gemma4 loading through the exact same load_aux_model path used in train_complextro.py"
    )
    parser.add_argument("--tokenizer-dir", required=True, help="Tokenizer/processor directory")
    parser.add_argument("--qwen-model-type", required=True, help="Model type passed to QwenImageTextEncoder, e.g. gemma4")
    parser.add_argument("--qwen-model-file", required=True, help="Model checkpoint path")
    parser.add_argument("--output", default=None, help="Output txt path")
    parser.add_argument("--device", default=None, help="Override device. Default uses Accelerator().device")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--enable-vram-offload", action="store_true")
    parser.add_argument("--vram-limit", type=float, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or f"tools/debug/gemma4_pipeline_path_report_{timestamp}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = []

    def log(title, content=""):
        report.append(f"\n===== {title} =====\n")
        if isinstance(content, str):
            report.append(content)
        else:
            report.append(json.dumps(content, ensure_ascii=False, indent=2))
        report.append("\n")

    try:
        import importlib
        import accelerate
        import torch
        from transformers import AutoProcessor
 
        from diffsynth.core import load_model
        from diffsynth.core.loader.file import load_state_dict
        from diffsynth.core.vram import AutoWrappedModule
        from diffsynth.configs.vram_management_module_maps import VRAM_MANAGEMENT_MODULE_MAPS, VERSION_CHECKER_MAPS
        from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder
        from diffsynth.utils.state_dict_converters.qwen_image_text_encoder import QwenImageTextEncoderStateDictConverter

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        accelerator = accelerate.Accelerator()
        torch_dtype = dtype_map[args.dtype]
        device = args.device if args.device is not None else accelerator.device
        device_str = str(device)
        qwen_model_type = args.qwen_model_type
        qwen_model_file = args.qwen_model_file
        qwen_tokenizer_dir = args.tokenizer_dir
        qwen_model_size = "2B" if qwen_model_type == "gemma4" else "0.8B"

        log("ARGS", vars(args))
        log("ACCELERATOR_DEVICE", device_str)
        log("ACCELERATOR_MIXED_PRECISION", str(getattr(accelerator.state, "mixed_precision", None)))
        log(
            "HARDCODED_PIPELINE_CALL",
            "self.pipe.text_encoder = load_aux_model(\n"
            "    QwenImageTextEncoder,\n"
            "    qwen_model_file,\n"
            "    config={\"model_type\": qwen_model_type, \"model_size\": qwen_model_size},\n"
            "    state_dict_converter=QwenImageTextEncoderStateDictConverter,\n"
            ")",
        )

        def resolve_module_map(model_class):
            def import_class(class_path: str):
                split = class_path.rfind(".")
                module_name, class_name = class_path[:split], class_path[split + 1:]
                return getattr(importlib.import_module(module_name), class_name)

            model_class_path = f"{model_class.__module__}.{model_class.__name__}"
            if model_class_path == "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder":
                return {model_class: AutoWrappedModule}
            if model_class_path in VERSION_CHECKER_MAPS:
                raw_map = VERSION_CHECKER_MAPS[model_class_path]()
                return {import_class(source): import_class(target) for source, target in raw_map.items()}
            if model_class_path not in VRAM_MANAGEMENT_MODULE_MAPS:
                raise KeyError(f"No VRAM management module map registered for {model_class_path}.")
            raw_map = VRAM_MANAGEMENT_MODULE_MAPS[model_class_path]
            return {import_class(source): import_class(target) for source, target in raw_map.items()}

        def load_aux_model(model_class, model_file, *, config=None, state_dict_converter=None):
            load_kwargs = {
                "config": config,
                "torch_dtype": torch.bfloat16,
                "device": device_str,
                "state_dict_converter": state_dict_converter,
            }
            if args.enable_vram_offload:
                vram_config = {
                    "offload_dtype": torch.bfloat16,
                    "offload_device": "cpu",
                    "onload_dtype": torch.bfloat16,
                    "onload_device": device_str,
                    "preparing_dtype": torch.bfloat16,
                    "preparing_device": device_str,
                    "computation_dtype": torch.bfloat16,
                    "computation_device": device_str,
                }
                load_kwargs["module_map"] = resolve_module_map(model_class)
                load_kwargs["vram_config"] = vram_config
                load_kwargs["vram_limit"] = args.vram_limit
            return load_model(model_class, model_file, **load_kwargs)

        raw_state_dict = load_state_dict(qwen_model_file, torch_dtype=torch_dtype, device=device_str)
        log("RAW_KEY_COUNT", str(len(raw_state_dict)))
        log("RAW_KEY_PREVIEW", "\n".join(list(raw_state_dict.keys())[:300]))

        converted_state_dict = QwenImageTextEncoderStateDictConverter(raw_state_dict)
        log("CONVERTED_KEY_COUNT", str(len(converted_state_dict)))
        log("CONVERTED_KEY_PREVIEW", "\n".join(list(converted_state_dict.keys())[:300]))

        probe_model = QwenImageTextEncoder(model_type=qwen_model_type, model_size=qwen_model_size)
        probe_keys = list(probe_model.state_dict().keys())
        log("MODEL_KEY_COUNT", str(len(probe_keys)))
        log("MODEL_KEY_PREVIEW", "\n".join(probe_keys[:300]))

        unexpected = sorted(set(converted_state_dict.keys()) - set(probe_keys))
        missing = sorted(set(probe_keys) - set(converted_state_dict.keys()))
        log("UNEXPECTED_KEY_COUNT", str(len(unexpected)))
        log("UNEXPECTED_KEYS_FULL", "\n".join(unexpected))
        log("MISSING_KEY_COUNT", str(len(missing)))
        log("MISSING_KEYS_FULL", "\n".join(missing))

        try:
            processor = AutoProcessor.from_pretrained(qwen_tokenizer_dir)
            log("PROCESSOR_LOAD", f"SUCCESS: {type(processor).__name__}")
            if hasattr(processor, "tokenizer"):
                log("TOKENIZER_LOAD", f"SUCCESS: {type(processor.tokenizer).__name__}")
        except Exception:
            log("PROCESSOR_LOAD_EXCEPTION", traceback.format_exc())

        try:
            model = load_aux_model(
                QwenImageTextEncoder,
                qwen_model_file,
                config={"model_type": qwen_model_type, "model_size": qwen_model_size},
                state_dict_converter=QwenImageTextEncoderStateDictConverter,
            )
            log("LOAD_RESULT", f"SUCCESS: {type(model).__name__}")
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
