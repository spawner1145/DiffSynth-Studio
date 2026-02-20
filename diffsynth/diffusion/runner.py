import os, math, ast, importlib, time, torch
from multiprocessing import Value
from tqdm import tqdm
from accelerate import Accelerator
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


def _build_optimizer(
    optimizer_type,
    trainable_params,
    learning_rate,
    weight_decay,
    optimizer_kwargs=None,
    optimizer_args=None,
):
    optimizer_type = "AdamW" if optimizer_type in [None, ""] else optimizer_type
    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    optimizer_kwargs.update(_parse_kv_args(optimizer_args))

    optimizer_type_lower = optimizer_type.lower()
    if optimizer_type_lower == "adamw":
        optimizer_kwargs.setdefault("weight_decay", weight_decay)

    if "." in optimizer_type:
        split = optimizer_type.rfind(".")
        module = importlib.import_module(optimizer_type[:split])
        class_name = optimizer_type[split + 1:]
        optimizer_class = _resolve_attr_case_insensitive(module, class_name)
    else:
        optimizer_class = _resolve_attr_case_insensitive(torch.optim, optimizer_type)

    optimizer = optimizer_class(trainable_params, lr=learning_rate, **optimizer_kwargs)
    return optimizer


def _build_scheduler(
    optimizer,
    lr_scheduler_type,
    total_steps,
    lr_warmup_steps=0,
    lr_scheduler_args=None,
):
    lr_scheduler_type = "constant" if lr_scheduler_type in [None, ""] else lr_scheduler_type
    lr_scheduler_kwargs = _parse_kv_args(lr_scheduler_args)

    if "." in lr_scheduler_type:
        split = lr_scheduler_type.rfind(".")
        module = importlib.import_module(lr_scheduler_type[:split])
        class_name = lr_scheduler_type[split + 1:]
        scheduler_class = _resolve_attr_case_insensitive(module, class_name)
        return scheduler_class(optimizer, **lr_scheduler_kwargs)

    lr_scheduler_type = lr_scheduler_type.lower()
    warmup_steps = max(0, int(lr_warmup_steps))
    total_steps = max(1, int(total_steps))

    def warmup_ratio(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    if lr_scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, **lr_scheduler_kwargs)

    if lr_scheduler_type == "constant_with_warmup":
        def lr_lambda(step):
            return warmup_ratio(step)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, **lr_scheduler_kwargs)

    if lr_scheduler_type == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return warmup_ratio(step)
            denom = max(1, total_steps - warmup_steps)
            progress = float(step - warmup_steps + 1) / float(denom)
            return max(0.0, 1.0 - progress)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, **lr_scheduler_kwargs)

    if lr_scheduler_type == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return warmup_ratio(step)
            denom = max(1, total_steps - warmup_steps)
            progress = min(1.0, float(step - warmup_steps + 1) / float(denom))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, **lr_scheduler_kwargs)

    try:
        scheduler_class = _resolve_attr_case_insensitive(torch.optim.lr_scheduler, lr_scheduler_type)
        return scheduler_class(optimizer, **lr_scheduler_kwargs)
    except AttributeError:
        pass

    raise ValueError(
        f"Unsupported lr_scheduler_type: {lr_scheduler_type}. Use built-in constant/constant_with_warmup/linear/cosine, "
        + "a torch.optim.lr_scheduler class name, or full class path."
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


def _normalize_log_with(log_with):
    if log_with is None:
        return []
    if isinstance(log_with, str):
        return [item.strip().lower() for item in log_with.split(",") if item.strip() != ""]
    return [str(item).strip().lower() for item in log_with if str(item).strip() != ""]


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
    log_with = None,
    logging_dir: str = None,
    tracker_project_name: str = "diffsynth-training",
    tracker_run_name: str = None,
    tracker_config: dict = None,
    log_every_n_steps: int = 1,
):
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
    log_with = _get_arg(args, "log_with", log_with)
    logging_dir = _get_arg(args, "logging_dir", logging_dir)
    tracker_project_name = _get_arg(args, "tracker_project_name", tracker_project_name)
    tracker_run_name = _get_arg(args, "tracker_run_name", tracker_run_name)
    log_every_n_steps = _get_arg(args, "log_every_n_steps", log_every_n_steps)
    num_workers = _get_arg(args, "dataset_num_workers", num_workers)
    save_steps = _get_arg(args, "save_steps", save_steps)
    save_epochs = _get_arg(args, "save_epochs", save_epochs)
    num_epochs = _get_arg(args, "num_epochs", num_epochs)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = _build_optimizer(
        optimizer_type,
        trainable_params,
        learning_rate,
        weight_decay,
        optimizer_kwargs=optimizer_kwargs,
        optimizer_args=optimizer_args,
    )

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
    collate_fn = _DatasetStateCollator(current_epoch, current_step, dataset, base_collate_fn)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=not bucket_batching_enabled,
        batch_size=dataloader_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    grad_accum_steps = max(1, int(getattr(accelerator, "gradient_accumulation_steps", 1)))
    num_processes = max(1, int(getattr(accelerator, "num_processes", 1)))
    num_update_steps_per_epoch = math.ceil(len(dataloader) / num_processes / grad_accum_steps)
    total_steps = max(1, num_update_steps_per_epoch * int(num_epochs))
    scheduler = _build_scheduler(
        optimizer,
        lr_scheduler_type=lr_scheduler_type,
        total_steps=total_steps,
        lr_warmup_steps=lr_warmup_steps,
        lr_scheduler_args=lr_scheduler_args,
    )

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
    global_step = 0
    log_every_n_steps = max(1, int(log_every_n_steps))
    
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
                loss = _compute_loss(model, dataset, data)
                accelerator.backward(loss)
                grad_norm = None
                if max_grad_norm is not None:
                    current_trainable_params = [param for param in model.parameters() if param.requires_grad]
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(current_trainable_params, max_grad_norm)
                elif show_grad_norm and accelerator.sync_gradients:
                    current_trainable_params = [param for param in model.parameters() if param.requires_grad]
                    grad_norm = _compute_grad_norm(current_trainable_params)
                optimizer.step()
                if accelerator.sync_gradients:
                    optimizer.zero_grad()
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss, epoch_id=epoch_id)
                    scheduler.step()
                    global_step += 1
                    epoch_update_steps += 1

                    lr = optimizer.param_groups[0].get("lr", None)
                    loss_item = loss.detach().float().item()
                    epoch_loss_sum += loss_item
                    update_duration = max(1e-12, time.perf_counter() - update_start_time)
                    update_start_time = time.perf_counter()
                    world_size = int(getattr(accelerator, "num_processes", 1))
                    local_batch_size = _infer_local_batch_size(data)
                    samples_per_update = local_batch_size * world_size * grad_accum_steps
                    samples_per_sec = samples_per_update / update_duration
                    postfix = {"loss": f"{loss_item:.4f}"}
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
