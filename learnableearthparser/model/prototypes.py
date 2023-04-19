import torch
from torch import nn
import numpy as np
import math

import learnableearthparser.utils.generate_shape as generate_shape
import learnableearthparser.utils.rotation as rotation

from ..utils import color as color

class Prototypes:

    def __initprotos__(self, datamodule):
        
        self.register_buffer("_protosrotation",
            torch.cat(
                [rotation._axis_angle_rotation("Y", (
                    self.hparams.decoders.scales.theta_L * math.pi * any(["rotY" in t for t in self.hparams.transformations]) * self.hparams.protos.init_with_fixed_y_rotation)
                    + torch.tensor(self.hparams.protos.init_with_different_y_rotation * math.pi * (k + 1) / (2 * max(self.hparams.K + 1, 1.))
                )).unsqueeze(0)
                for k in range(self.hparams.K)], 0
            )
        )
        if self.hparams.protos.name == "points":
            self._protos = nn.Parameter(torch.stack(
                [proto for proto in self.sample_proto()], dim=0
            ))
        elif self.hparams.protos.name == "superquadrics":
            pass
        else:
            raise ValueError

        if self.hparams.protos.learn_specific_scale:
            self._protosscale = torch.tensor(self.hparams.protos.init).view(-1, 1, 3)
            if self._protosscale.size(0) != self.hparams.K:
                self._protosscale = self._protosscale.repeat(self.hparams.K, 1, 1)[:self.hparams.K]
            self._protosscale = self._protosscale + self.hparams.protos.noise_specific_scale * (2*torch.rand_like(self._protosscale) - 1)
            self._protosscale = nn.Parameter(self._protosscale)
        else:
            self.register_buffer("_protosscale", torch.tensor(self.hparams.protos.init).view(1, 1, 3))

        if self.hparams.distance == "xyzk":
            self._protosfeat = nn.Parameter(torch.tensor([
                np.percentile(datamodule.train_dataset.data.intensity.numpy(), q) for q in np.linspace(0, 100, 2*self.hparams.K+1)[1::2]
            ]).unsqueeze(-1).float())

        if self.hparams.distance == "xyzk":
            self.register_buffer("lambda_xyz_feat", 25.6 * torch.tensor(
                3*[1. / self.hparams.data.max_xy] + [self.hparams.lambda_xyz_feat]))
        elif self.hparams.distance == "xyz":
            self.register_buffer("lambda_xyz_feat", 25.6 * torch.tensor(
                3*[1. / self.hparams.data.max_xy]))
        else:
            raise NotImplementedError

    def sample_proto(self):
        return [
                torch.from_numpy(getattr(generate_shape, self.hparams.protos.shape[k % len(self.hparams.protos.shape)])(self.hparams.protos.points))
            for k in range(self.hparams.K)
        ]

    def get_protosfeat(self):
        return self._protosfeat

    def get_protosscale(self):
        return self.hparams.protos.specific_scale * self.hparams.data.max_xy * torch.exp(self._protosscale)

    @torch.profiler.record_function(f"GET P")
    def get_protos(self, points=None):
        if self.hparams.protos.name == "points":
            return torch.matmul(self._protos * self.get_protosscale(), self._protosrotation)
        elif self.hparams.protos.name == "superquadrics":
            return self.get_protosscale()
        else:
            raise ValueError