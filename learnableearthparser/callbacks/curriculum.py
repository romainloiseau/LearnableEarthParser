import torch
import copy

from .base import DTICallback

from .outhtml import OutHTML

class Curriculum(DTICallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.monitor = self.myhparams.monitor
        self.patience = self.myhparams.patience
        self.last_value = None
        self.count = 0

        self.order = self.myhparams.order

        self.warmup = 0

        self.to_activate = None

        if self.myhparams.mode == "min":
            self.mode = -1
        elif self.myhparams.mode == "max":
            self.mode = 1
        else:
            raise ValueError(f"Argument 'mode' should be in ['min', 'max']\t'{self.myhparams.mode}' is invalid.")
    
    def on_train_start(self, trainer, pl_module):
        for transformation in self.order:
            if "decay" not in transformation and "protos" not in transformation:
                pl_module.activated_transformations[transformation] = False

        for param_group in pl_module.optimizers().param_groups:
            if (param_group["name"] in self.order) or (("protos" == param_group["name"]) and ("protos" in self.order)):
                param_group['lr'] *= 0

        for decoder in pl_module.decoders:
            if decoder in self.order:
                torch.nn.init.zeros_(
                    getattr(pl_module, "decoders")[decoder][-1].weight,
                )
                torch.nn.init.zeros_(
                    getattr(pl_module, "decoders")[decoder][-1].bias,
                )

    def on_train_epoch_end(self, trainer, pl_module):
        if (len(self.order) != 0) and (self.to_activate is None):
            if self.last_value is None:
                if self.monitor in trainer.callback_metrics:
                    self.last_value = trainer.callback_metrics[self.monitor].item()
            elif self.mode*trainer.callback_metrics[self.monitor].item() > self.mode*self.last_value:
                self.last_value = trainer.callback_metrics[self.monitor].item()
                self.count = 0
            else:
                self.count += 1

                if self.count >= self.patience:
                    self.activate_transformation(trainer, pl_module)
                    self.last_value = None
                    self.count = 0

        for param_group in pl_module.optimizers().param_groups:
            pl_module.log(f'Learning_rate/{param_group["name"]}', param_group["lr"], on_step=False, on_epoch=True)
        
    def activate_transformation(self, trainer, pl_module):
        self.to_activate, self.order = self.order[0], self.order[1:]

        self.send_message(f"Activating {self.to_activate}", trainer.current_epoch)
        if self.myhparams.save_ckpt_at_activation:
            trainer.save_checkpoint(f"checkpoints/epoch-{trainer.current_epoch}_{self.to_activate}.ckpt")

        if self.myhparams.log_html_at_activation:
            for callback in trainer.callbacks:
                if isinstance(callback, OutHTML):
                    callback.do_out_html(
                        trainer, pl_module,
                        title=f"Activating {self.to_activate} at epoch {trainer.current_epoch}",
                        name=f"epoch={trainer.current_epoch}_{self.to_activate}"
                    )

        if "decay" not in self.to_activate:
            if "protos" not in self.to_activate:
                pl_module.activated_transformations[self.to_activate] = True
            self.warmup = copy.copy(self.myhparams.warmup_batch)

            for param_group in pl_module.optimizers().param_groups[1:]:
                if param_group["name"] not in self.order:
                    if "protos" == param_group["name"]:
                        param_group['lr'] = pl_module.hparams.protos.optim.lr*self.myhparams.warmup_intensity
                    else:
                        param_group['lr'] = pl_module.hparams.optim.lr*self.myhparams.warmup_intensity
                        
        else:
            decay = float(self.to_activate.split("_")[1])
            for param_group in pl_module.optimizers().param_groups:
                param_group['lr'] = param_group['lr'] / decay
        
    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):

        if self.warmup > 0:
            self.warmup -= 1
            for param_group in pl_module.optimizers().param_groups[1:]:
                if param_group["name"] not in self.order:
                    if "protos" == param_group["name"]:
                        param_group['lr'] += (1. - self.myhparams.warmup_intensity) * pl_module.hparams.protos.optim.lr / self.myhparams.warmup_batch
                    else:
                        param_group['lr'] += (1. - self.myhparams.warmup_intensity) * pl_module.hparams.optim.lr / self.myhparams.warmup_batch
                                    
        if self.warmup == 0:
            self.to_activate = None