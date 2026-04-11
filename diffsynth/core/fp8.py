"""FP8 / Transformer Engine helpers for training.

Provides utilities to:
- Replace nn.Linear with TE Linear for FP8 computation
- Wrap modules with TE-aware gradient checkpointing
- Create FP8 autocast contexts with configurable recipes
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn


ModuleFilter = Callable[[str, nn.Module], bool]

DEFAULT_TE_LINEAR_EXCLUDES = (
    "time_text_embed",
    "text_pool_proj",
    "norm_out.linear",
)


def _import_te():
    try:
        import transformer_engine.pytorch as te
    except ImportError as exc:
        raise ImportError(
            "Transformer Engine is required for FP8 training. "
            "Install transformer_engine and use an FP8-capable GPU (e.g. H100)."
        ) from exc
    return te


def _default_filter(_name: str, _module: nn.Module) -> bool:
    return True


def make_te_linear_filter(exclude_modules: Optional[Sequence[str]] = None) -> ModuleFilter:
    """Build a prefix-based filter for selecting modules to replace with TE Linear."""
    excluded = tuple(
        str(item).strip()
        for item in (exclude_modules or DEFAULT_TE_LINEAR_EXCLUDES)
        if str(item).strip()
    )

    def _filter(name: str, _module: nn.Module) -> bool:
        return not any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in excluded
        )

    return _filter


def replace_linears_with_te(
    root: nn.Module,
    *,
    module_filter: Optional[ModuleFilter] = None,
    exclude_modules: Optional[Sequence[str]] = None,
) -> list[str]:
    """Replace selected nn.Linear modules in-place with TE Linear."""
    te = _import_te()
    if module_filter is not None and exclude_modules is not None:
        raise ValueError("Pass either module_filter or exclude_modules, not both.")
    module_filter = module_filter or make_te_linear_filter(exclude_modules)
    replaced: list[str] = []

    def _replace(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and module_filter(full_name, child):
                te_linear = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                )
                te_linear.to(device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    te_linear.weight.copy_(child.weight)
                    if child.bias is not None and te_linear.bias is not None:
                        te_linear.bias.copy_(child.bias)
                setattr(parent, child_name, te_linear)
                replaced.append(full_name)
                continue
            _replace(child, full_name)

    _replace(root)
    return replaced


def has_te_linear(root: nn.Module) -> bool:
    """Check whether any module in the tree is a TE Linear."""
    try:
        te = _import_te()
    except ImportError:
        return False
    return any(isinstance(m, te.Linear) for m in root.modules())


def wrap_modules_for_te_checkpointing(
    root: nn.Module,
    *,
    module_filter: Optional[ModuleFilter] = None,
    use_reentrant: bool = True,
    checkpoint_in_eval: bool = False,
) -> list[str]:
    """Wrap selected modules so their forward runs under te.checkpoint()."""
    te = _import_te()
    module_filter = module_filter or _default_filter
    wrapped: list[str] = []

    for module_name, module in root.named_modules():
        if module_name == "":
            continue
        if not module_filter(module_name, module):
            continue
        if getattr(module, "_te_checkpoint_wrapped", False):
            continue

        original_forward = module.forward
        signature = inspect.signature(original_forward)

        def checkpointed_forward(
            *args,
            __module=module,
            __orig=original_forward,
            __sig=signature,
            **kwargs,
        ):
            if (not checkpoint_in_eval and not __module.training) or not torch.is_grad_enabled():
                return __orig(*args, **kwargs)

            bound = __sig.bind(*args, **kwargs)
            bound.apply_defaults()
            ordered_args = tuple(bound.arguments[param_name] for param_name in __sig.parameters)
            return te.checkpoint(__orig, *ordered_args, use_reentrant=use_reentrant)

        module._te_original_forward = original_forward
        module._te_checkpoint_wrapped = True
        module.forward = checkpointed_forward
        wrapped.append(module_name)

    return wrapped


def unwrap_te_checkpointing(root: nn.Module) -> list[str]:
    """Restore original forward methods for modules wrapped by TE checkpointing."""
    restored: list[str] = []
    for module_name, module in root.named_modules():
        if module_name == "":
            continue
        original_forward = getattr(module, "_te_original_forward", None)
        if original_forward is None:
            continue
        module.forward = original_forward
        delattr(module, "_te_original_forward")
        delattr(module, "_te_checkpoint_wrapped")
        restored.append(module_name)
    return restored


def create_fp8_autocast_context(
    enabled: bool = True,
    fp8_format: str = "HYBRID",
    amax_history_len: int = 16,
    amax_compute_algo: str = "max",
) -> Callable:
    """Create an FP8 autocast context manager factory.

    Returns a callable that returns a context manager for FP8 autocast.
    When enabled=False, returns contextlib.nullcontext.
    """
    if not enabled:
        return contextlib.nullcontext

    te = _import_te()
    from transformer_engine.common.recipe import DelayedScaling, Format

    fp8_format_map = {
        "HYBRID": Format.HYBRID,
        "E4M3": Format.E4M3,
        "E5M2": Format.E5M2,
    }
    fp8_format_name = str(fp8_format).upper()
    if fp8_format_name not in fp8_format_map:
        raise ValueError(
            f"Unsupported FP8 format: {fp8_format_name}. "
            f"Choose from: {list(fp8_format_map.keys())}"
        )
    recipe = DelayedScaling(
        fp8_format=fp8_format_map[fp8_format_name],
        amax_history_len=int(amax_history_len),
        amax_compute_algo=str(amax_compute_algo),
    )

    def fp8_autocast_ctx():
        return te.autocast(enabled=True, recipe=recipe)

    return fp8_autocast_ctx
