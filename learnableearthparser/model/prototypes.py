import torch
from torch import nn
import numpy as np
import math

import learnableearthparser.utils.generate_shape as generate_shape
import learnableearthparser.utils.rotation as rotation

from ..utils import color as color
from ..fast_sampler import fast_sample_on_batch
from ..utils import superquadrics as sq

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
        elif self.hparams.protos.name in ["superquadrics", "cube_diff"]:
            pass
        elif self.hparams.protos.name == "superquadrics_diff":
            self._protos = nn.Parameter(torch.zeros((self.hparams.K, 2)))
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
        elif self.hparams.protos.name == "cube_diff":
            if self.training:
                _protos = 2. * torch.rand((self.hparams.K, self.hparams.protos.points, 3), device=self.device) - 1.
            else:
                _protos = torch.stack(torch.meshgrid((torch.linspace(-1, 1, 6, device=self.device), torch.linspace(-1, 1, 6, device=self.device), torch.linspace(-1, 1, 6, device=self.device)))).flatten(1, 3).T.unsqueeze(0).repeat(self.hparams.K, 1, 1)
            return torch.matmul(_protos * self.get_protosscale(), self._protosrotation)
        elif self.hparams.protos.name == "superquadrics":
            return self.get_protosscale()
        elif self.hparams.protos.name == "superquadrics_diff":
            epsilons = torch.sigmoid(self._protos)*1.1 + 0.4

            scales = self.get_protosscale()[:, 0]

            if self.training:
                etas, omegas = fast_sample_on_batch(scales.unsqueeze(0).detach().cpu().numpy(), epsilons.unsqueeze(0).detach().cpu().numpy(), self.hparams.protos.points)
                etas, omegas = etas[0], omegas[0]
            else:
                from ..utils.icosphere import generate_icosphere
                v, faces = generate_icosphere(2)

                etas, omegas = torch.asin(v[:, 2] / torch.norm(v, dim=1)), torch.atan2(v[:, 1], v[:, 0])
                etas, omegas = etas.view(1, -1).repeat(self.hparams.K, 1).numpy(), omegas.view(1, -1).repeat(self.hparams.K, 1).numpy()

            # Make sure we don't get nan for gradients
            etas[etas == 0] += 1e-6
            omegas[omegas == 0] += 1e-6

            # Move to tensors
            etas = scales.new_tensor(etas)
            omegas = scales.new_tensor(omegas)

            protos = sq.get_superquadrics_diff(epsilons, scales, etas, omegas)

            return protos
        else:
            raise ValueError