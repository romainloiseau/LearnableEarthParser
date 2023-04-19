import torch
import spconv.pytorch as spconv

import logging
import yaml
import hydra
import os
import numpy as np
import copy
import os.path as osp
import pytorch_lightning as pl
from omegaconf import DictConfig

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import multidti3d

logger = logging.getLogger(__name__)

def update_log(curr_string, to_add):
    logger.info(to_add)
    return curr_string + to_add

@hydra.main(config_path="configs", config_name="postproc")
def main(cfg: DictConfig) -> None:

    root = cfg.postproc.root
    metric = cfg.postproc.metric
    thresh = cfg.postproc.thresh
    
    for xp in os.listdir(root):
        if osp.exists(osp.join(root, xp, "checkpoints")):
            
            logger.info("")
            logger.info(f"{xp}")

            root_xp = osp.join(root, xp)
            ckpts = osp.join(root_xp, "checkpoints")

            epoch = 0
            for ckpt in os.listdir(ckpts):
                if "step" in ckpt and ckpt.endswith(".ckpt"):
                    this_epoch = int(ckpt.split("=")[-1].split(".")[0])
                    if this_epoch > epoch:
                        epoch = this_epoch
                        last_ckpt = osp.join(ckpts, ckpt)

            overrides = osp.join(root_xp, ".hydra", "overrides.yaml")
            with open(overrides, "r") as stream:
                overrides = yaml.safe_load(stream)

            cfg = hydra.compose(
                config_name="defaults",
                overrides=overrides + [
                    "hydra.searchpath=[file://../../../EarthParserDataset/configs]",
                    f"model.callbacks.outhtml.points_per_scene=0",
                    f"mode=test",
                ]
            )
            del cfg.model.callbacks.outhtml, cfg.model.callbacks.earlystopping, cfg.model.callbacks.curriculum, cfg.model.callbacks.protocounter

            cfg.model.load_weights = last_ckpt

            logger.info(f"{xp}\tcfg loaded : {last_ckpt}")

            if cfg.seed != 0: pl.seed_everything(cfg.seed)
            #logger.info("\n" + OmegaConf.to_yaml(cfg))

            cfg.data._target_ = f"EarthParserDataset.{cfg.data._target_}"
            datamodule = hydra.utils.instantiate(cfg.data)
            datamodule.setup("test")

            logger.info(f"{xp}\tDatamodule loaded : {datamodule}")

            model = hydra.utils.instantiate(
                cfg.model,
                _recursive_=False,
            )
            model.__initprotos__(datamodule)

            if cfg.model.load_weights != "":
                logger.info(f"{xp}\tLoading weights from {cfg.model.load_weights}")
                model.load_state_dict(torch.load(cfg.model.load_weights)["state_dict"], strict=False)

            model = model.eval()

            out_string = ""

            trainer = multidti3d.trainers.get_trainer(cfg)
            out = trainer.test(model, datamodule=datamodule, verbose=False)
            del trainer

            out_string = update_log(out_string, f'Chamfer = {1000*out[0]["Losses/chamfer/test"]:.4f}\tChamfer4D = {1000*out[0]["Losses/chamfer4D/test"] if "Losses/chamfer4D/test" in out[0].keys() else 0:.4f}\tIoU = {100*out[0]["IoU/test"]:.2f}\n')

            curr_chamfer = out[0][f"Losses/{metric}/test"]
            remove = []
            model.remove_proto = []

            while True:

                chamfers = []
                chamfers4D = []
                ious = []
                removes = []
                for remove_proto in range(1, 1 + cfg.model.K):
                    if remove_proto not in remove:
                        model.remove_proto = list(dict.fromkeys(remove + [remove_proto]))

                        trainer = multidti3d.trainers.get_trainer(cfg)
                        out = trainer.test(model, datamodule=datamodule, verbose=False)

                        chamfers.append(out[0]["Losses/chamfer/test"])
                        chamfers4D.append(out[0]["Losses/chamfer4D/test"] if "Losses/chamfer4D/test" in out[0].keys() else 0)
                        ious.append(out[0]["IoU/test"])
                        removes.append(copy.deepcopy(model.remove_proto))

                        out_string = update_log(out_string, f'\tw/o {removes[-1]}\tChamfer = {1000*chamfers[-1]:.4f}\tChamfer4D = {1000*chamfers4D[-1]:.4f}\tIoU = {100*ious[-1]:.2f}\n')

                        del trainer, out
                
                if metric == "chamfer4D":
                    mini = np.argmin(chamfers4D)
                    new_chamfer = chamfers4D[mini]
                elif metric == "chamfer":
                    mini = np.argmin(chamfers)
                    new_chamfer = chamfers[mini]
                
                if (new_chamfer - curr_chamfer) / curr_chamfer < thresh:
                    remove = copy.deepcopy(removes[mini])
                    curr_chamfer = copy.deepcopy(new_chamfer)
                    
                    out_string = update_log(out_string, f"\nRemoving prototype {remove}\tChamfer = {1000*chamfers[mini]:.4f}\tChamfer4D = {1000*chamfers4D[mini]:.4f}\tIoU = {100*ious[mini]:.2f}\n")
                    
                else:
                    out_string = update_log(out_string, "STOP ! Chamfer loss doesn't decrease anymore !")
                    break
                    
            logger.info(out_string)
            with open(f"{xp}.txt", "w") as f:
                f.write(out_string)

            logger.info("{xp}\tDone !")
            del model, datamodule, cfg

if __name__ == '__main__':
    main()