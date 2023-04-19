import numpy as np
import matplotlib.pyplot as plt
import torch
from .base import DTICallback


import itertools


class IoU(DTICallback):
    
    def __init__(self, *args, **kwargs):
        self.confusion_matrix_iou = {}
        self.confusion_matrix_acc = {}
        self.aris = {}

        super().__init__(*args, **kwargs)

    def reset(self, tag):
        self.aris[tag] = []

        self.confusion_matrix_iou[tag] = torch.zeros((self.confusion_matrix_size[tag] * self.n_classes, ), dtype=torch.int64)
        self.confusion_matrix_acc[tag] = torch.zeros((self.confusion_matrix_size[tag] * self.n_classes, ), dtype=torch.int64)

    def delete(self, tag):
        if tag in self.confusion_matrix_iou.keys():
            del self.confusion_matrix_iou[tag]
            del self.confusion_matrix_acc[tag]
            del self.aris[tag]
    
    @torch.profiler.record_function(f"CONFMAT")
    def update_confusion_matrix(self, outputs, batch, tag):
        if "y_pred" in outputs:
            to_confmat = self.confusion_matrix_size[tag] * batch.point_y.flatten() + outputs["y_pred"].flatten()
            unique, counts = torch.unique(to_confmat.flatten(), return_counts=True)
            self.confusion_matrix_iou[tag][unique.detach().cpu().long()] += counts.detach().cpu()

    @torch.no_grad()
    def compute_metrics(self, trainer, pl_module):
        confmat = {tag: cm.reshape(self.n_classes, self.confusion_matrix_size[tag]).detach().cpu().numpy() for tag, cm in self.confusion_matrix_iou.items()}

        key = "train" if "train" in confmat.keys() else ("test" if "test" in confmat.keys() else "val")

        if self.ignore_index_0:
            self.best_assign = 1 + np.argmax(confmat[key][1:], axis=0)
        else:
            self.best_assign = np.argmax(confmat[key], axis=0)

        confmat = {tag: np.vstack([cm[:, self.best_assign == c].sum(axis=1) for c in range(self.n_classes)]).transpose() for tag, cm in confmat.items()}
        
        protoid = ["-".join(np.where(self.best_assign==c)[0].astype(str)) for c in range(self.n_classes)]

        for tag, cm in confmat.items():
            self.log_iou(pl_module, tag, cm)

            if self.do_greedy_step(trainer.current_epoch):
                if len(protoid) <= 20 and tag != "test":
                    trainer.logger.experiment.add_image(
                        f"iou_assigned/{tag}", self.image_confusion_matrix(cm, pl_module.hparams.data.class_names, protoid),
                        global_step=trainer.current_epoch, dataformats='HWC'
                    )
                    
        for tag, aris in self.aris.items():
            if len(aris) > 0:
                pl_module.log(f'ARI/{tag}', np.mean(aris), on_step=False, on_epoch=True)

    def log_iou(self, pl_module, tag, cm):
        if self.ignore_index_0:
            thiscm = cm[1:, 1:]
        else:
            thiscm = cm

        intersection = np.diag(thiscm)

        pl_module.log(f'Acc/{tag}', intersection.sum() / thiscm.sum(), on_step=False, on_epoch=True)

        union = thiscm.sum(0) + thiscm.sum(1) - intersection

        scores = intersection / union
        scores[union == 0] = 0

        pl_module.log(f'IoU/{tag}', scores.mean(), on_step=False, on_epoch=True)

    @torch.no_grad()
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int) -> None:
        self.update_confusion_matrix(outputs, batch, tag="train")

    @torch.no_grad()
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.update_confusion_matrix(outputs, batch, tag="val")
    
    @torch.no_grad()
    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.update_confusion_matrix(outputs, batch, tag="test")

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.reset("val")

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.reset("test")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.reset("train")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.compute_metrics(trainer, pl_module)
        self.delete("train")
        self.delete("val")
        self.delete("test")

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.compute_metrics(trainer, pl_module)
        self.delete("train")
        self.delete("val")
        self.delete("test")

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_start(trainer, pl_module)

        self.confusion_matrix_size = {"train": self.K, "val": self.K, "test": self.K * self.K_points}
        
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        super().on_train_start(trainer, pl_module)

        self.confusion_matrix_size = {"train": self.K, "val": self.K, "test": self.K * self.K_points}

        if hasattr(trainer.datamodule.train_dataset, "data"):
            figure = plt.figure()
            bins = np.arange(self.n_classes + 1) - .5
            plt.hist(trainer.datamodule.train_dataset.data.point_y.flatten().numpy(), bins=bins)
            plt.xlim((-.5, self.n_classes-.5))
            plt.xlabel("Class")
            plt.yscale("log")
            plt.ylabel("Number of points")
            plt.xticks(np.arange(self.n_classes), pl_module.hparams.data.class_names, rotation=30)
            s, (width, height) = figure.canvas.print_to_buffer()
            plt.tight_layout()
            plt.clf()
            plt.close(figure)
            del figure
            trainer.logger.experiment.add_image(
                f"Class_distribution/train", np.fromstring(s, np.uint8).reshape((height, width, 4)),
                global_step=trainer.current_epoch, dataformats='HWC'
            )

        return

    def image_confusion_matrix(self, cm, cm_classes, cm_protoid=None):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        """

        n_samples = cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm.astype('float') / n_samples)

        figure = plt.figure()

        plt.imshow((cm - cm.min()) / (cm.max() - cm.min()) if cm.max() != cm.min() else cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        plt.tick_params(labelright=False, right=True)
        plt.xticks(np.arange(cm.shape[1]), np.arange(cm.shape[1]) if cm_protoid is None else cm_protoid)
        plt.yticks(np.arange(cm.shape[0]), cm_classes, rotation=60)

        threshold = cm.min() + .5 * (cm.max() - cm.min())
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            cmfloat = np.around(
                100 * cm[i, j], decimals=1 if 100 * cm[i, j] >= 10 else 2) if cm[i, j] != 0 else ""
            plt.text(j, i, cmfloat, horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted prototype')

        s, (width, height) = figure.canvas.print_to_buffer()
        plt.clf()
        plt.close(figure)
        del figure
        return np.fromstring(s, np.uint8).reshape((height, width, 4))