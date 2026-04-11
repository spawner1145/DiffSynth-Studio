import contextlib
import os, math, ast, importlib, time, torch
import argparse
from typing import Any, Callable, Optional, Tuple
from multiprocessing import Value
from collections import deque
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import Optimizer
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _get_arg(args, key, default):
    if args is None:
        return default
    return getattr(args, key, default)


def _parse_kv_args(kv_args):
    if kv_args is None:
        return {}
    if isinstance(kv_args, dict):
        return dict(kv_args)
    if isinstance(kv_args, str):
        kv_args = [kv_args]
    parsed = {}
    for raw_arg in kv_args:
        arg = str(raw_arg).strip()
        if arg == "":
            continue
        if "=" not in arg:
            raise ValueError(f"Invalid argument '{arg}'. Expected format: key=value")
        key, value = arg.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "":
            raise ValueError(f"Invalid argument '{arg}'. Key cannot be empty")
        try:
            value = ast.literal_eval(value)
        except Exception:
            pass
        parsed[key] = value
    return parsed


def _resolve_attr_case_insensitive(module, attr_name):
    if hasattr(module, attr_name):
        return getattr(module, attr_name)
    lowered = attr_name.lower()
    for name in dir(module):
        if name.lower() == lowered:
            return getattr(module, name)
    raise AttributeError(f"Cannot find {attr_name} in module {module.__name__}")


def _infer_mup_dim(model) -> Optional[float]:
    if model is None:
        return None

    dim_attr_names = (
        "mup_dim",
        "hidden_size",
        "model_dim",
        "width",
        "dim",
        "embed_dim",
        "text_embed_dim",
        "d_model",
    )
    container_attr_names = (
        "model",
        "pipe",
        "dit",
        "unet",
        "text_encoder",
        "transformer",
        "encoder",
        "decoder",
    )

    def _normalize_dim(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value) if value > 0 else None
        try:
            value = int(value)
            return float(value) if value > 0 else None
        except Exception:
            return None

    checked = set()
    queue = [model]
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in checked:
            continue
        checked.add(id(current))

        if isinstance(current, dict):
            for name in dim_attr_names:
                if name in current:
                    dim = _normalize_dim(current.get(name))
                    if dim is not None:
                        return dim
        else:
            for name in dim_attr_names:
                if hasattr(current, name):
                    dim = _normalize_dim(getattr(current, name))
                    if dim is not None:
                        return dim

        config = getattr(current, "config", None)
        if config is not None and id(config) not in checked:
            queue.append(config)

        for name in container_attr_names:
            if hasattr(current, name):
                child = getattr(current, name)
                if isinstance(child, (list, tuple)):
                    queue.extend(list(child))
                else:
                    queue.append(child)

    return None


def _scale_learning_rate_mup(learning_rate: float, mup_dim: float, mup_base_dim: float) -> float:
    if learning_rate is None:
        return learning_rate
    if mup_dim is None or mup_dim <= 0:
        raise ValueError(f"mup_dim must be > 0, got {mup_dim}")
    if mup_base_dim is None or mup_base_dim <= 0:
        raise ValueError(f"mup_base_dim must be > 0, got {mup_base_dim}")
    return float(learning_rate) * math.sqrt(float(mup_base_dim) / float(mup_dim))


def get_optimizer(args, trainable_params) -> Tuple[str, str, object]:
    # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, PagedAdamW, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, AdEMAMix8bit, PagedAdEMAMix8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, Adafactor"

    optimizer_type = getattr(args, "optimizer_type", None)
    if getattr(args, "use_8bit_adam", False):
        assert (
            not getattr(args, "use_lion_optimizer", False)
        ), "both option use_8bit_adam and use_lion_optimizer are specified"
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_8bit_adam and optimizer_type are specified"
        optimizer_type = "AdamW8bit"

    elif getattr(args, "use_lion_optimizer", False):
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_lion_optimizer and optimizer_type are specified"
        optimizer_type = "Lion"

    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"
    optimizer_type = optimizer_type.lower()

    if getattr(args, "fused_backward_pass", False):
        assert (
            optimizer_type == "Adafactor".lower()
        ), "fused_backward_pass currently only works with optimizer_type Adafactor"
        assert (
            getattr(args, "gradient_accumulation_steps", 1) == 1
        ), "fused_backward_pass does not work with gradient_accumulation_steps > 1"

    optimizer_kwargs = {}
    optimizer_args = getattr(args, "optimizer_args", None)
    if optimizer_args is not None and len(optimizer_args) > 0:
        for arg in optimizer_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value

    lr = getattr(args, "learning_rate", None)
    optimizer = None
    optimizer_class = None

    if optimizer_type == "Lion".lower():
        try:
            import lion_pytorch
        except ImportError:
            raise ImportError("No lion_pytorch")
        print(f"use Lion optimizer | {optimizer_kwargs}")
        optimizer_class = lion_pytorch.Lion
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("8bit".lower()):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes")

        if optimizer_type == "AdamW8bit".lower():
            print(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = bnb.optim.AdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "SGDNesterov8bit".lower():
            print(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
            if "momentum" not in optimizer_kwargs:
                print(
                    f"8-bit SGD with Nesterov must be with momentum, set momentum to 0.9"
                )
                optimizer_kwargs["momentum"] = 0.9

            optimizer_class = bnb.optim.SGD8bit
            optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

        elif optimizer_type == "Lion8bit".lower():
            print(f"use 8-bit Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.Lion8bit
            except AttributeError:
                raise AttributeError(
                    "No Lion8bit. The version of bitsandbytes installed seems to be old. Please install 0.38.0 or later."
                )
        elif optimizer_type == "PagedAdamW8bit".lower():
            print(f"use 8-bit PagedAdamW optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedAdamW8bit
            except AttributeError:
                raise AttributeError(
                    "No PagedAdamW8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later."
                )
        elif optimizer_type == "PagedLion8bit".lower():
            print(f"use 8-bit Paged Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedLion8bit
            except AttributeError:
                raise AttributeError(
                    "No PagedLion8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later."
                )

        if optimizer_class is not None:
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "PagedAdamW".lower():
        print(f"use PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandby")
        try:
            optimizer_class = bnb.optim.PagedAdamW
        except AttributeError:
            raise AttributeError(
                "No PagedAdamW. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later."
            )
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "PagedAdamW32bit".lower():
        print(f"use 32-bit PagedAdamW optimizer | {optimizer_kwargs}")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes")
        try:
            optimizer_class = bnb.optim.PagedAdamW32bit
        except AttributeError:
            raise AttributeError(
                "No PagedAdamW32bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later."
            )
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov".lower():
        print(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            print(
                f"SGD with Nesterov must be with momentum, set momentum to 0.9"
            )
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type.startswith("DAdapt".lower()) or optimizer_type == "Prodigy".lower():
        # check lr and lr_count, and logger.info warning
        actual_lr = lr
        lr_count = 1
        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            lrs = set()
            actual_lr = trainable_params[0].get("lr", actual_lr)
            for group in trainable_params:
                lrs.add(group.get("lr", actual_lr))
            lr_count = len(lrs)

        if actual_lr <= 0.1:
            print(
                f"learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0: lr={actual_lr}"
            )
            print("recommend option: lr=1.0")
        if lr_count > 1:
            print(
                f"when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), only the first one will take effect: lr={actual_lr}"
            )

        if optimizer_type.startswith("DAdapt".lower()):
            # DAdaptation family
            # check dadaptation is installed
            try:
                import dadaptation
                import dadaptation.experimental as experimental
            except ImportError:
                raise ImportError("No dadaptation")

            # set optimizer
            if optimizer_type == "DAdaptation".lower() or optimizer_type == "DAdaptAdamPreprint".lower():
                optimizer_class = experimental.DAdaptAdamPreprint
                print(f"use D-Adaptation AdamPreprint optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdaGrad".lower():
                optimizer_class = dadaptation.DAdaptAdaGrad
                print(f"use D-Adaptation AdaGrad optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdam".lower():
                optimizer_class = dadaptation.DAdaptAdam
                print(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdan".lower():
                optimizer_class = dadaptation.DAdaptAdan
                print(f"use D-Adaptation Adan optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdanIP".lower():
                optimizer_class = experimental.DAdaptAdanIP
                print(f"use D-Adaptation AdanIP optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptLion".lower():
                optimizer_class = dadaptation.DAdaptLion
                print(f"use D-Adaptation Lion optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptSGD".lower():
                optimizer_class = dadaptation.DAdaptSGD
                print(f"use D-Adaptation SGD optimizer | {optimizer_kwargs}")
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")

            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
        else:
            # Prodigy
            # check Prodigy is installed
            try:
                import prodigyopt
            except ImportError:
                raise ImportError("No Prodigy")

            print(f"use Prodigy optimizer | {optimizer_kwargs}")
            optimizer_class = prodigyopt.Prodigy
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
        # 引数を確認して適宜補正する
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            print(
                f"set relative_step to True because warmup_init is True"
            )
            optimizer_kwargs["relative_step"] = True
        print(f"use Adafactor optimizer | {optimizer_kwargs}")

        if optimizer_kwargs["relative_step"]:
            print(f"relative_step is true")
            if lr != 0.0:
                print(f"learning rate is used as initial_lr")
            args.learning_rate = None

            if type(trainable_params) == list and type(trainable_params[0]) == dict:
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

                if has_group_lr:
                    print(f"unet_lr and text_encoder_lr are ignored")
                    args.unet_lr = None
                    args.text_encoder_lr = None

            if args.lr_scheduler != "adafactor":
                print(f"use adafactor_scheduler")
            args.lr_scheduler = f"adafactor:{lr}"

            lr = None
        else:
            if args.max_grad_norm != 0.0:
                print(
                    f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0"
                )
            if args.lr_scheduler != "constant_with_warmup":
                print(f"constant_with_warmup will be good")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                print(f"clip_threshold=1.0 will be good")

        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        print(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("schedulefree".lower()):
        try:
            import schedulefree as sf
        except ImportError:
            raise ImportError("No schedulefree")

        if optimizer_type == "RAdamScheduleFree".lower():
            optimizer_class = sf.RAdamScheduleFree
            print(f"use RAdamScheduleFree optimizer | {optimizer_kwargs}")
        elif optimizer_type == "AdamWScheduleFree".lower():
            optimizer_class = sf.AdamWScheduleFree
            print(f"use AdamWScheduleFree optimizer | {optimizer_kwargs}")
        elif optimizer_type == "SGDScheduleFree".lower():
            optimizer_class = sf.SGDScheduleFree
            print(f"use SGDScheduleFree optimizer | {optimizer_kwargs}")
        else:
            optimizer_class = None

        if optimizer_class is not None:
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    if optimizer is None:
        case_sensitive_optimizer_type = getattr(args, "optimizer_type", "AdamW")  # not lower
        print(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

        if "." not in case_sensitive_optimizer_type:  # from torch.optim
            optimizer_module = torch.optim
        else:  # from other library
            values = case_sensitive_optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            case_sensitive_optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    # for logging
    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    if hasattr(optimizer, "train") and callable(optimizer.train):
        # make optimizer as train mode before training for schedulefree optimizer. the optimizer will be in eval mode in sampling and saving.
        optimizer.train()

    return optimizer_name, optimizer_args, optimizer


def get_optimizer_train_eval_fn(optimizer: Optimizer, args: argparse.Namespace) -> Tuple[Callable, Callable]:
    if not is_schedulefree_optimizer(optimizer, args):
        # return dummy func
        return lambda: None, lambda: None

    # get train and eval functions from optimizer
    train_fn = optimizer.train
    eval_fn = optimizer.eval

    return train_fn, eval_fn


def is_schedulefree_optimizer(optimizer: Optimizer, args: argparse.Namespace) -> bool:
    optimizer_type = getattr(args, "optimizer_type", "")
    return str(optimizer_type).lower().endswith("schedulefree".lower())  # or args.optimizer_schedulefree_wrapper


def get_dummy_scheduler(optimizer: Optimizer) -> Any:
    # dummy scheduler for schedulefree optimizer. supports only empty step(), get_last_lr() and optimizers.
    # this scheduler is used for logging only.
    # this isn't be wrapped by accelerator because of this class is not a subclass of torch.optim.lr_scheduler._LRScheduler
    class DummyScheduler:
        def __init__(self, optimizer: Optimizer):
            self.optimizer = optimizer

        def step(self):
            pass

        def get_last_lr(self):
            return [group["lr"] for group in self.optimizer.param_groups]

    return DummyScheduler(optimizer)


def get_scheduler_fix(args, optimizer: Optimizer, num_processes: int):
    """
    Unified API to get any scheduler from its name.
    """
    # if schedulefree optimizer, return dummy scheduler
    if is_schedulefree_optimizer(optimizer, args):
        return get_dummy_scheduler(optimizer)

    name = getattr(args, "lr_scheduler", "constant")
    num_training_steps = getattr(args, "max_train_steps", None)
    num_training_steps = num_training_steps * num_processes if num_training_steps is not None else None
    num_warmup_steps: Optional[int] = (
        int(getattr(args, "lr_warmup_steps", 0) * num_training_steps)
        if isinstance(getattr(args, "lr_warmup_steps", 0), float) and num_training_steps is not None
        else getattr(args, "lr_warmup_steps", 0)
    )
    num_decay_steps: Optional[int] = (
        int(getattr(args, "lr_decay_steps", 0) * num_training_steps)
        if isinstance(getattr(args, "lr_decay_steps", 0), float) and num_training_steps is not None
        else getattr(args, "lr_decay_steps", 0)
    )
    num_stable_steps = (num_training_steps - num_warmup_steps - num_decay_steps) if num_training_steps is not None else 0
    num_cycles = getattr(args, "lr_scheduler_num_cycles", 1)
    power = getattr(args, "lr_scheduler_power", 1.0)
    timescale = getattr(args, "lr_scheduler_timescale", 1.0)
    min_lr_ratio = getattr(args, "lr_scheduler_min_lr_ratio", None)

    lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
    lr_scheduler_args = getattr(args, "lr_scheduler_args", None)
    if lr_scheduler_args is not None and len(lr_scheduler_args) > 0:
        for arg in lr_scheduler_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            lr_scheduler_kwargs[key] = value

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(f"{name} does not require num_warmup_steps. Set None or 0.")
        return return_vals

    # using any lr_scheduler from other library
    lr_scheduler_type = getattr(args, "lr_scheduler_type", "")
    if lr_scheduler_type:
        print(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
        if "." not in lr_scheduler_type:  # default to use torch.optim
            lr_scheduler_module = torch.optim.lr_scheduler
        else:
            values = lr_scheduler_type.split(".")
            lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
            lr_scheduler_type = values[-1]
        lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
        lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
        return wrap_check_needless_num_warmup_steps(lr_scheduler)

    if name.startswith("adafactor"):
        assert (
            type(optimizer) == transformers.optimization.Adafactor
        ), f"adafactor scheduler must be used with Adafactor optimizer"
        initial_lr = float(name.split(":")[1])
        return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

    if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
        name = DiffusersSchedulerType(name)
        schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
        return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

    # All other schedulers require num_warmup_steps
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires num_warmup_steps, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, timescale=timescale, **lr_scheduler_kwargs)

    # All other schedulers require num_training_steps
    if num_training_steps is None:
        raise ValueError(f"{name} requires num_training_steps, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            **lr_scheduler_kwargs,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power, **lr_scheduler_kwargs
        )

    if name == SchedulerType.COSINE_WITH_MIN_LR:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles / 2,
            min_lr_rate=min_lr_ratio,
            **lr_scheduler_kwargs,
        )

    # these schedulers do not require num_decay_steps
    if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **lr_scheduler_kwargs,
        )

    # All other schedulers require `num_decay_steps`
    if num_decay_steps is None:
        raise ValueError(f"{name} requires num_decay_steps, please provide that argument.")
    if name == SchedulerType.WARMUP_STABLE_DECAY:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_stable_steps=num_stable_steps,
            num_decay_steps=num_decay_steps,
            num_cycles=num_cycles / 2,
            min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
            **lr_scheduler_kwargs,
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_decay_steps=num_decay_steps,
        **lr_scheduler_kwargs,
    )


def _compute_loss(model, dataset, data):
    def _all_tensors_with_same_shape(items):
        if len(items) == 0:
            return False
        if not all(isinstance(item, torch.Tensor) for item in items):
            return False
        ref = items[0]
        for item in items[1:]:
            if item.shape != ref.shape or item.dtype != ref.dtype:
                return False
        return True

    def _try_merge_batch(samples):
        if len(samples) == 0:
            return None
        if _all_tensors_with_same_shape(samples):
            return torch.stack(samples, dim=0)
        if all(isinstance(sample, dict) for sample in samples):
            keys = set(samples[0].keys())
            for sample in samples[1:]:
                if set(sample.keys()) != keys:
                    return None
            merged = {}
            for key in keys:
                values = [sample[key] for sample in samples]
                if _all_tensors_with_same_shape(values):
                    merged[key] = torch.stack(values, dim=0)
                else:
                    merged[key] = values
            return merged
        return None

    if isinstance(data, list):
        merged_data = _try_merge_batch(data)
        if merged_data is not None:
            if dataset.load_from_cache:
                return model({}, inputs=merged_data)
            return model(merged_data)
        losses = []
        for sample in data:
            if dataset.load_from_cache:
                sample_loss = model({}, inputs=sample)
            else:
                sample_loss = model(sample)
            losses.append(sample_loss)
        return torch.stack(losses).mean()
    if dataset.load_from_cache:
        return model({}, inputs=data)
    return model(data)


def _infer_local_batch_size(data):
    if isinstance(data, list):
        return max(1, len(data))
    if isinstance(data, torch.Tensor):
        if data.ndim >= 1:
            return max(1, int(data.shape[0]))
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return max(1, int(value.shape[0]))
            if isinstance(value, (list, tuple)):
                return max(1, len(value))
    return 1


def _compute_grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            grad = param.grad.detach()
            total += grad.norm(2).item() ** 2
    return total ** 0.5


def _compute_layer_grad_norms(named_parameters):
    layer_totals = {}
    for name, param in named_parameters:
        if param.grad is None:
            continue
        layer_name = name.rsplit(".", 1)[0] if "." in name else name
        grad_norm_sq = param.grad.detach().norm(2).item() ** 2
        layer_totals[layer_name] = layer_totals.get(layer_name, 0.0) + grad_norm_sq
    return {layer_name: total ** 0.5 for layer_name, total in layer_totals.items()}


def _normalize_log_with(log_with):
    if log_with is None:
        return []
    if isinstance(log_with, str):
        return [item.strip().lower() for item in log_with.split(",") if item.strip() != ""]
    return [str(item).strip().lower() for item in log_with if str(item).strip() != ""]


def _is_builtin_lr_scheduler_name(name: Optional[str]) -> bool:
    if name is None:
        return True
    name_str = str(name)
    if name_str == "":
        return True
    if name_str.startswith("adafactor"):
        return True
    if name_str == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
        return True
    try:
        SchedulerType(name_str)
        return True
    except Exception:
        return False


class _DatasetStateCollator:
    def __init__(self, current_epoch, current_step, dataset, base_collate_fn):
        self.current_epoch = current_epoch
        self.current_step = current_step
        self.dataset = dataset
        self.base_collate_fn = base_collate_fn

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_dataset = worker_info.dataset
        else:
            worker_dataset = self.dataset

        if hasattr(worker_dataset, "set_current_epoch"):
            worker_dataset.set_current_epoch(self.current_epoch.value)
        if hasattr(worker_dataset, "set_current_step"):
            worker_dataset.set_current_step(self.current_step.value)
        return self.base_collate_fn(examples)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    save_epochs: int = 1,
    num_epochs: int = 1,
    args = None,
    batch_size: int = 1,
    optimizer_type: str = "adamw",
    optimizer_kwargs: dict = None,
    optimizer_args = None,
    lr_scheduler_type: str = "constant",
    lr_scheduler_args = None,
    lr_warmup_steps: int = 0,
    max_grad_norm: float = None,
    show_grad_norm: bool = True,
    log_layer_grad_norms: bool = False,
    mup_scale: bool = False,
    mup_base_dim: float = 1.0,
    mup_dim: Optional[float] = None,
    log_with = None,
    logging_dir: str = None,
    tracker_project_name: str = "diffsynth-training",
    tracker_run_name: str = None,
    tracker_config: dict = None,
    log_every_n_steps: int = 1,
):
    if args is None:
        args = argparse.Namespace()

    learning_rate = _get_arg(args, "learning_rate", learning_rate)
    weight_decay = _get_arg(args, "weight_decay", weight_decay)
    batch_size = _get_arg(args, "batch_size", _get_arg(args, "train_batch_size", batch_size))
    optimizer_type = _get_arg(args, "optimizer_type", optimizer_type)
    optimizer_args = _get_arg(args, "optimizer_args", optimizer_args)
    lr_scheduler_type = _get_arg(args, "lr_scheduler", _get_arg(args, "lr_scheduler_type", lr_scheduler_type))
    lr_scheduler_type = _get_arg(args, "lr_scheduler_type", lr_scheduler_type)
    lr_scheduler_args = _get_arg(args, "lr_scheduler_args", lr_scheduler_args)
    lr_warmup_steps = _get_arg(args, "lr_warmup_steps", lr_warmup_steps)
    max_grad_norm = _get_arg(args, "max_grad_norm", max_grad_norm)
    show_grad_norm = _get_arg(args, "show_grad_norm", show_grad_norm)
    log_layer_grad_norms = _get_arg(args, "log_layer_grad_norms", log_layer_grad_norms)
    mup_scale = _get_arg(args, "mup_scale", mup_scale)
    mup_base_dim = _get_arg(args, "mup_base_dim", mup_base_dim)
    mup_dim = _get_arg(args, "mup_dim", mup_dim)
    log_with = _get_arg(args, "log_with", log_with)
    logging_dir = _get_arg(args, "logging_dir", logging_dir)
    tracker_project_name = _get_arg(args, "tracker_project_name", tracker_project_name)
    tracker_run_name = _get_arg(args, "tracker_run_name", tracker_run_name)
    log_every_n_steps = _get_arg(args, "log_every_n_steps", log_every_n_steps)
    num_workers = _get_arg(args, "dataset_num_workers", num_workers)
    save_steps = _get_arg(args, "save_steps", save_steps)
    save_epochs = _get_arg(args, "save_epochs", save_epochs)
    num_epochs = _get_arg(args, "num_epochs", num_epochs)

    if mup_scale:
        if mup_dim is None:
            mup_dim = _infer_mup_dim(model)
        if mup_dim is None:
            raise ValueError("mup_scale=True but failed to infer mup_dim from model. Please pass mup_dim.")
        learning_rate = _scale_learning_rate_mup(learning_rate, mup_dim, mup_base_dim)
        args.learning_rate = learning_rate

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if optimizer_args is not None:
        args.optimizer_args = optimizer_args
    if optimizer_type is not None:
        args.optimizer_type = optimizer_type
    if optimizer_kwargs is not None:
        # keep backward-compatible path; kwargs take precedence via optimizer_args when provided
        if args.optimizer_args is None:
            args.optimizer_args = []
        for key, value in optimizer_kwargs.items():
            args.optimizer_args.append(f"{key}={value}")

    if not hasattr(args, "optimizer_args"):
        args.optimizer_args = None
    if not hasattr(args, "use_8bit_adam"):
        args.use_8bit_adam = False
    if not hasattr(args, "use_lion_optimizer"):
        args.use_lion_optimizer = False
    if not hasattr(args, "fused_backward_pass"):
        args.fused_backward_pass = False

    if not hasattr(args, "gradient_accumulation_steps"):
        args.gradient_accumulation_steps = int(getattr(accelerator, "gradient_accumulation_steps", 1))
    if not hasattr(args, "max_grad_norm"):
        args.max_grad_norm = max_grad_norm if max_grad_norm is not None else 0.0
    if not hasattr(args, "lr_scheduler"):
        args.lr_scheduler = lr_scheduler_type
    if not hasattr(args, "learning_rate"):
        args.learning_rate = learning_rate

    optimizer_name, optimizer_args_log, optimizer = get_optimizer(args, trainable_params)

    dataset_seed = _get_arg(args, "seed", 0)
    if hasattr(dataset, "set_seed"):
        dataset.set_seed(dataset_seed)
    if hasattr(dataset, "set_batch_size"):
        dataset.set_batch_size(int(batch_size))

    bucket_batching_enabled = bool(getattr(dataset, "bucket_enabled_batching", False))
    dataloader_batch_size = 1 if bucket_batching_enabled else int(batch_size)
    base_collate_fn = (lambda x: x[0]) if dataloader_batch_size == 1 else (lambda x: x)
    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    # Attach shared values so persistent workers can sync epoch/step in __getitem__.
    dataset._shared_epoch_value = current_epoch
    dataset._shared_step_value = current_step
    collate_fn = _DatasetStateCollator(current_epoch, current_step, dataset, base_collate_fn)
    use_persistent_workers = num_workers > 0 and hasattr(dataset, '_sync_shared_state')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=not bucket_batching_enabled,
        batch_size=dataloader_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent_workers,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    grad_accum_steps = max(1, int(getattr(accelerator, "gradient_accumulation_steps", 1)))
    num_processes = max(1, int(getattr(accelerator, "num_processes", 1)))
    num_update_steps_per_epoch = math.ceil(len(dataloader) / num_processes / grad_accum_steps)
    total_steps = max(1, num_update_steps_per_epoch * int(num_epochs))
    if not hasattr(args, "lr_scheduler_args"):
        args.lr_scheduler_args = lr_scheduler_args
    if not hasattr(args, "lr_scheduler_type"):
        if _is_builtin_lr_scheduler_name(lr_scheduler_type):
            args.lr_scheduler_type = ""
        else:
            args.lr_scheduler_type = lr_scheduler_type
    if not hasattr(args, "lr_warmup_steps"):
        args.lr_warmup_steps = lr_warmup_steps
    if not hasattr(args, "lr_decay_steps"):
        args.lr_decay_steps = 0
    if not hasattr(args, "lr_scheduler_num_cycles"):
        args.lr_scheduler_num_cycles = 1
    if not hasattr(args, "lr_scheduler_power"):
        args.lr_scheduler_power = 1.0
    if not hasattr(args, "lr_scheduler_timescale"):
        args.lr_scheduler_timescale = 1.0
    if not hasattr(args, "lr_scheduler_min_lr_ratio"):
        args.lr_scheduler_min_lr_ratio = None
    if not hasattr(args, "max_train_steps"):
        args.max_train_steps = total_steps

    scheduler = get_scheduler_fix(args, optimizer, num_processes)

    enabled_trackers = _normalize_log_with(log_with)
    tb_writer = None
    wandb_run = None
    if accelerator.is_main_process and len(enabled_trackers) > 0:
        if "tensorboard" in enabled_trackers:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = logging_dir if logging_dir is not None else os.path.join("runs", tracker_project_name)
            tb_writer = SummaryWriter(log_dir=tb_dir)
        if "wandb" in enabled_trackers:
            import wandb
            wandb_run = wandb.init(
                project=tracker_project_name,
                name=tracker_run_name,
                dir=logging_dir,
                config=tracker_config,
                reinit=True,
            )

    model.to(device=accelerator.device)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # 恢复优化器状态
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
    if resume_from_checkpoint is not None:
        state_dir = resume_from_checkpoint.replace(".safetensors", "_optimizer_state")
        if os.path.exists(state_dir):
            if accelerator.is_main_process:
                print(f"检测到对应的优化器状态目录 {state_dir}，正在加载...")
            accelerator.load_state(state_dir)
            if accelerator.is_main_process:
                print(f"优化器状态加载成功！")

    # FP8 TE autocast context
    fp8_te_enabled = _get_arg(args, "fp8_te_enabled", False)
    if fp8_te_enabled:
        from ..core.fp8 import create_fp8_autocast_context
        fp8_autocast_ctx = create_fp8_autocast_context(
            enabled=True,
            fp8_format=_get_arg(args, "fp8_te_format", "HYBRID"),
            amax_history_len=_get_arg(args, "fp8_te_amax_history_len", 16),
            amax_compute_algo=_get_arg(args, "fp8_te_amax_compute_algo", "max"),
        )
        if accelerator.is_main_process:
            print(f"FP8 TE autocast enabled: format={_get_arg(args, 'fp8_te_format', 'HYBRID')}, "
                  f"amax_history_len={_get_arg(args, 'fp8_te_amax_history_len', 16)}, "
                  f"amax_compute_algo={_get_arg(args, 'fp8_te_amax_compute_algo', 'max')}")
    else:
        fp8_autocast_ctx = contextlib.nullcontext

    global_step = 0
    log_every_n_steps = max(1, int(log_every_n_steps))
    progress_loss_keys = _get_arg(args, "progress_loss_keys", None)
    if progress_loss_keys is None:
        progress_loss_keys = ["loss_ema"]  #progress_loss_keys=["loss","loss_ma50"] 或字符串：progress_loss_keys="loss,loss_ma50"
    elif isinstance(progress_loss_keys, str):
        progress_loss_keys = [k.strip() for k in progress_loss_keys.split(",") if k.strip()]
    else:
        progress_loss_keys = list(progress_loss_keys)
    trend_loss_ema = None
    trend_loss_ema_beta = 0.98
    trend_loss_window = deque(maxlen=50)
    
    for epoch_id in range(num_epochs):
        current_epoch.value = int(epoch_id)
        if hasattr(dataset, "set_current_epoch"):
            dataset.set_current_epoch(epoch_id)
        optimizer.zero_grad()
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch_id + 1}/{num_epochs}")
        epoch_start_time = time.perf_counter()
        update_start_time = epoch_start_time
        epoch_loss_sum = 0.0
        epoch_update_steps = 0
        for data in progress_bar:
            current_step.value = int(global_step)
            with accelerator.accumulate(model):
                with fp8_autocast_ctx():
                    loss = _compute_loss(model, dataset, data)
                accelerator.backward(loss)
                grad_norm = None
                layer_grad_norms = None
                if max_grad_norm is not None:
                    current_trainable_params = [param for param in model.parameters() if param.requires_grad]
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(current_trainable_params, max_grad_norm)
                elif show_grad_norm and accelerator.sync_gradients:
                    current_trainable_params = [param for param in model.parameters() if param.requires_grad]
                    grad_norm = _compute_grad_norm(current_trainable_params)
                if log_layer_grad_norms and accelerator.sync_gradients:
                    unwrapped_model = accelerator.unwrap_model(model)
                    current_trainable_named_params = [
                        (name, param) for name, param in unwrapped_model.named_parameters() if param.requires_grad
                    ]
                    layer_grad_norms = _compute_layer_grad_norms(current_trainable_named_params)
                optimizer.step()
                if accelerator.sync_gradients:
                    optimizer.zero_grad()
                    global_step += 1
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss, epoch_id=epoch_id, global_step=global_step)
                    scheduler.step()
                    epoch_update_steps += 1

                    lr = optimizer.param_groups[0].get("lr", None)
                    loss_item = loss.detach().float().item()
                    if trend_loss_ema is None:
                        trend_loss_ema = loss_item
                    else:
                        trend_loss_ema = trend_loss_ema_beta * trend_loss_ema + (1.0 - trend_loss_ema_beta) * loss_item
                    trend_loss_window.append(loss_item)
                    trend_loss_mean = sum(trend_loss_window) / max(1, len(trend_loss_window))
                    epoch_loss_sum += loss_item
                    update_duration = max(1e-12, time.perf_counter() - update_start_time)
                    update_start_time = time.perf_counter()
                    world_size = int(getattr(accelerator, "num_processes", 1))
                    local_batch_size = _infer_local_batch_size(data)
                    samples_per_update = local_batch_size * world_size * grad_accum_steps
                    samples_per_sec = samples_per_update / update_duration
                    loss_entries = {
                        "loss": f"{loss_item:.4f}",
                        "loss_ema": f"{trend_loss_ema:.4f}",
                        "loss_ma50": f"{trend_loss_mean:.4f}",
                    }
                    postfix = {k: loss_entries[k] for k in progress_loss_keys if k in loss_entries}
                    if lr is not None:
                        postfix["lr"] = f"{lr:.2e}"
                    if grad_norm is not None:
                        try:
                            grad_norm_item = grad_norm.detach().float().item()
                        except AttributeError:
                            grad_norm_item = float(grad_norm)
                        postfix["grad"] = f"{grad_norm_item:.3f}"
                    progress_bar.set_postfix(postfix)

                    if accelerator.is_main_process and global_step % log_every_n_steps == 0:
                        metrics = {
                            "train/loss": loss_item,
                            "train/loss_ema": float(trend_loss_ema),
                            "train/loss_ma50": float(trend_loss_mean),
                            "train/epoch": float(epoch_id),
                            "train/global_step": float(global_step),
                            "train/progress": float(global_step) / float(total_steps),
                            "train/step_time_sec": float(update_duration),
                            "train/samples_per_sec": float(samples_per_sec),
                        }
                        if lr is not None:
                            metrics["train/lr"] = float(lr)
                        if grad_norm is not None:
                            metrics["train/grad_norm"] = float(grad_norm_item)
                        extra_loss_metrics = getattr(model, "latest_loss_metrics", None)
                        if isinstance(extra_loss_metrics, dict):
                            for key, value in extra_loss_metrics.items():
                                if key == "total_loss":
                                    continue
                                metrics[f"train/{key}"] = float(value)
                        if layer_grad_norms is not None:
                            for layer_name, layer_grad_norm in layer_grad_norms.items():
                                metrics[f"grad_norm_layers/{layer_name.replace('.', '/')}"] = float(layer_grad_norm)
                        if tb_writer is not None:
                            for key, value in metrics.items():
                                tb_writer.add_scalar(key, value, global_step)
                        if wandb_run is not None:
                            wandb_run.log(metrics, step=global_step)
        if accelerator.is_main_process and epoch_update_steps > 0:
            epoch_duration = max(1e-12, time.perf_counter() - epoch_start_time)
            epoch_metrics = {
                "train/epoch_loss_mean": float(epoch_loss_sum / epoch_update_steps),
                "train/epoch_duration_sec": float(epoch_duration),
                "train/epoch_steps": float(epoch_update_steps),
                "train/epoch_samples_per_sec": float((epoch_update_steps * int(batch_size) * int(getattr(accelerator, "num_processes", 1)) * grad_accum_steps) / epoch_duration),
            }
            if tb_writer is not None:
                for key, value in epoch_metrics.items():
                    tb_writer.add_scalar(key, value, global_step)
            if wandb_run is not None:
                wandb_run.log(epoch_metrics, step=global_step)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id, save_epochs=save_epochs)

    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    model_logger.on_training_end(accelerator, model, save_steps, epoch_id=num_epochs - 1)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    num_workers = _get_arg(args, "dataset_num_workers", num_workers)
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
