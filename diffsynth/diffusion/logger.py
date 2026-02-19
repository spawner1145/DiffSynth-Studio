import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, checkpoint_name="model"):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        if checkpoint_name is None or str(checkpoint_name).strip() == "":
            checkpoint_name = "model"
        self.checkpoint_name = str(checkpoint_name)
        self.num_steps = 0


    def _checkpoint_file_name(self, epoch_id=None, step_id=None):
        epoch_num = 0 if epoch_id is None else int(epoch_id) + 1
        step_num = self.num_steps if step_id is None else int(step_id)
        return f"{self.checkpoint_name}-e{epoch_num}-s{step_num}.safetensors"


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        epoch_id = kwargs.get("epoch_id", None)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, self._checkpoint_file_name(epoch_id=epoch_id, step_id=self.num_steps))


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id, save_epochs=1):
        if save_epochs is None:
            save_epochs = 1
        save_epochs = max(1, int(save_epochs))
        if (int(epoch_id) + 1) % save_epochs != 0:
            return
        self.save_model(accelerator, model, self._checkpoint_file_name(epoch_id=epoch_id, step_id=self.num_steps))


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, epoch_id=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, self._checkpoint_file_name(epoch_id=epoch_id, step_id=self.num_steps))


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
