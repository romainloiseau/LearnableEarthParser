import torch
import spconv.pytorch as spconv

import logging
import yaml
import hydra
import os
import os.path as osp
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import multidti3d

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="test")
def main(cfg: DictConfig) -> None:

    html = f"""
    <html>
    <head>
    <style>
    table {{
    font-family: arial, sans-serif;
    border-collapse: collapse;
    width: 100%;
    }}
    td, th {{
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
    }}
    tr:nth-child(even) {{
    background-color: #dddddd;
    }}
    </style>
    </head>
    <body>
    <h1>Results</h1>
    <table>
    <tr>
        <th>Experiment</th>
        <th>Chamfer</th>
        <th>mIoU</th>
        <th>Acc</th>
        <th>RGB</th>
        <th>Intensity</th>
        <th>Ground truth</th>
        <th>Prediction</th>
        <th>Instance pred</th>
        <th>Rec intensity</th>
        <th>Rec prediction</th>
        <th>Protos</th>
        <th>Protos labels</th>
        <th>Report</th>
    </tr>
    """

    root = cfg.test.root    

    for xp in sorted(os.listdir(root)):
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

            cfg.model.load_weights = last_ckpt

            logger.info(f"{xp}\tcfg loaded : {last_ckpt}")

            if cfg.seed != 0: pl.seed_everything(cfg.seed)

            cfg.data._target_ = f"EarthParserDataset.{cfg.data._target_}"
            datamodule = hydra.utils.instantiate(cfg.data)
            datamodule.setup("train")
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

            trainer = multidti3d.trainers.get_trainer(cfg)

            trainer.test(model, datamodule=datamodule)

            if "fit" in cfg.model.load_weights:
                modelname = model.hparams.load_weights.split("fit/")[-1].split("/")[0]
            else:
                modelname = model.hparams.load_weights.split("outputs/")[-1].split("/")[0]
                
            html += f"""
            <tr>
                <td>{cfg.experiment}</td>
                <td>{1000*trainer.logged_metrics['Losses/chamfer/test']:.3f}</td>
                <td>{100*trainer.logged_metrics['IoU/test']:.3f}</td>
                <td>{100*trainer.logged_metrics['Acc/test']:.3f}</td>
                <td><img width=100px src="{cfg.data.name}_{modelname}_input_rgb_0.png"/></td>
                <td><img width=100px src="{cfg.data.name}_{modelname}_input_intensity_0.png"/></td>
                <td><img width=100px src="{cfg.data.name}_{modelname}_input_point_y_0.png"/></td>
                <td><img width=100px src="{cfg.data.name}_{modelname}_input_point_y_pred_0.png"/></td>
                <td><img width=100px src="{cfg.data.name}_{modelname}_input_point_inst_pred_0.png"/></td>
                <td><img width=100px src="{cfg.data.name}_{modelname}_reconstruction_intensity_0.png"/></td>
                <td><img width=100px src="{cfg.data.name}_{modelname}_reconstruction_point_y_0.png"/></td>
                <td><img width=100px src="{cfg.data.name}_{modelname}_prototypes_rgb_0.png"/></td>
                <td><img width=100px src="{cfg.data.name}_{modelname}_prototypes_point_y_0.png"/></td>
                <td><a href="report_{cfg.data.name}_{modelname}_on_test_end.html">report_{cfg.data.name}_{modelname}_on_test_end.html</a></td>
            </tr>
            """

            logger.info("{xp}\tDone !")
            del trainer, model, datamodule, cfg

    html += """
    </body>
    </html>
    """

    with open(f"index.html", "w") as f:
        f.write(html)

if __name__ == '__main__':
    main()