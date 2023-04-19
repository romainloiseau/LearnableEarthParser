import numpy as np
import matplotlib.pyplot as plt
import torch
from .base import DTICallback

class ProtoCounter(DTICallback):

    def on_train_epoch_start(self, *args, **kwargs):
        self.count_assignments = torch.zeros((self.K, ))
        self.arange_k = torch.arange(self.K).unsqueeze(-1)

    @torch.profiler.record_function(f"ProtoCounter")
    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs) -> None:
        with torch.no_grad():
            equals = outputs["choice"].flatten().detach().cpu() == self.arange_k
            self.count_assignments += equals.sum(-1)

    def on_train_epoch_end(self, trainer, pl_module):
        count_assignments = self.K * self.count_assignments / self.count_assignments.sum()

        if self.do_greedy_step(trainer.current_epoch):
            figure = plt.figure()
            bins = np.arange(count_assignments.size(0) + 1) - .5
            centroids = (bins[1:] + bins[:-1]) / 2
            plt.hist(centroids, bins=bins,
                weights=count_assignments.numpy())
            plt.plot(bins, 1+0*bins, color="black")
            plt.ylim((0.0001, 10))
            plt.xlim((-.5, self.K-.5))
            plt.yscale("log")
            plt.xlabel("Prototype")
            plt.ylabel("Proportion *K")
            s, (width, height) = figure.canvas.print_to_buffer()
            plt.clf()
            plt.close(figure)
            del figure
            trainer.logger.experiment.add_image(
                f"assignments", np.fromstring(s, np.uint8).reshape((height, width, 4)),
                global_step=trainer.current_epoch, dataformats='HWC'
            )