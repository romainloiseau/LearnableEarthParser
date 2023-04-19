import logging

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import spconv.pytorch as spconv
import torch

plt.switch_backend('agg')

import learnableearthparser

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="defaults")
def main(cfg) -> None:

    # Fixing random seed
    if cfg.seed != 0: pl.seed_everything(cfg.seed)

    # Loading dataset
    cfg.data._target_ = f"EarthParserDataset.{cfg.data._target_}"
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(cfg.mode)

    # Instantiating model
    model = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )
    model.__initprotos__(datamodule)

    # Loading weights
    if cfg.model.load_weights != "":
        logger.info("Loading weights from %s", cfg.model.load_weights)
        model.load_state_dict(torch.load(cfg.model.load_weights)["state_dict"], strict=False)

    # Instantiating trainer
    trainer = learnableearthparser.trainers.get_trainer(cfg)

    # Running training
    getattr(trainer, cfg.mode)(model, datamodule=datamodule)

if __name__ == '__main__':
    main()