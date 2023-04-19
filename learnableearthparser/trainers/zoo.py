import pytorch_lightning as pl
from torch.profiler import ProfilerActivity
from omegaconf import  OmegaConf
import torch
import hydra

def get_trainer(cfg):
    logger = pl.loggers.TensorBoardLogger(".", "", "", log_graph=cfg.model.log_trace, default_hp_metric=False)

    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.model.callbacks.values()]
    
    if cfg.profile:
        profiler = pl.profiler.PyTorchProfiler(
            with_stack=False,
            record_shapes=False,
            profile_memory=cfg.profile==2,
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=10, warmup=1, active=3)
        )
    else:
        profiler=None

    return pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        profiler=profiler
    )