from tqdm.auto import tqdm

from types import SimpleNamespace
from pytorch_lightning.callbacks import Callback

class DTICallback(Callback):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.myhparams = SimpleNamespace(**kwargs)

    def send_message(self, message, epoch=0):
        tqdm.write(f"Epoch {epoch}:\t{message}")

    def do_greedy_step(self, epoch):
        return (epoch % int(self.log_every_n_epochs)) == 0

    def generate_global_params(self, pl_module):
        self.n_instances = max(pl_module.hparams.data.n_max, pl_module.hparams.S)
        
        self.n_classes = len(pl_module.hparams.data.class_names)
        self.K = pl_module.hparams.K
        self.K_points = pl_module.hparams.protos.points
        self.L = pl_module.hparams.S
        self.log_every_n_epochs = pl_module.hparams.log_every_n_epochs

        self.ignore_index_0 = pl_module.hparams.data.ignore_index_0

        self.max_xy = pl_module.hparams.data.max_xy
        self.max_z = pl_module.hparams.data.max_z

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.generate_global_params(pl_module)
        return super().on_train_start(trainer, pl_module)
        
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.generate_global_params(pl_module)
        return super().on_validation_start(trainer, pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.generate_global_params(pl_module)
        return super().on_test_start(trainer, pl_module)