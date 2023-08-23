import numpy as np
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn

import torch.nn.functional as F

import spconv.pytorch as spconv
import torch_scatter
from collections import OrderedDict

from .logging import LoggingModel
from .decoders import Decoders
from .prototypes import Prototypes
from .voxel import conv as VL
from .point import encoders as PE
from .point import decoders as PD

from ..utils import chamfer_distance

class BaseModel(pl.LightningModule, LoggingModel, Decoders, Prototypes):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.register_buffer("normalizer", torch.tensor(
            [[self.hparams.data.max_xy, self.hparams.data.max_xy, self.hparams.data.max_z]]))

        self.register_buffer("normalize_translation_L", torch.tensor(self.hparams.decoders.scales.translation_L))

        self.__initmodel__(*args, **kwargs)
        self.__initmetrics__()
        self.__compute_nparams__()

    def compute_features(self, batch):
        return 2 * batch.pos / self.normalizer - 1.0

    def __compute_nparams__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def __initmetrics__(self):
        return

    def __initmodel__(self, *args, **kwargs):
        self.N_voxels_tuple = (
            math.ceil(self.hparams.data.max_xy / self.hparams.encoder.first_res),
            math.ceil(self.hparams.data.max_xy / self.hparams.encoder.first_res),
            math.ceil(self.hparams.data.max_xy / self.hparams.encoder.first_res)
        )

        self.register_buffer("voxelizer", torch.tensor([
            self.hparams.encoder.first_res, self.hparams.encoder.first_res,
            self.hparams.data.max_z * self.hparams.encoder.first_res / self.hparams.data.max_xy
        ]))

        self.register_buffer("N_voxels", torch.tensor(
            self.N_voxels_tuple).unsqueeze(0).to(torch.int32))

        self.thin2coarse = int(2**(self.hparams.encoder.n_pools))

        self.register_buffer("tile_size", torch.tensor([
            self.hparams.encoder.first_res * self.thin2coarse,
            self.hparams.encoder.first_res * self.thin2coarse,
            self.hparams.encoder.first_res * self.thin2coarse * self.hparams.data.max_z / self.hparams.data.max_xy
        ]))

        self.__initnets__()

    def __initnets__(self):
        voxel_encoder = []

        for i in range(self.hparams.encoder.n_pools):
            voxel_encoder.append(
                VL.ConvLayer(
                    self.hparams.encoder.point_encoder[-1] if i == 0 else self.hparams.encoder.voxel_encoder[i-1],
                    self.hparams.encoder.voxel_encoder[i],
                    norm=self.hparams.normalization_layer,
                    key=f"{i}_conv"
                )
            )
            voxel_encoder.append(
                VL.DownConvLayer(
                    self.hparams.encoder.voxel_encoder[i],
                    self.hparams.encoder.voxel_encoder[i],
                    kernel_size=2,#if (i != self.hparams.n_pools-1) else 4,
                    stride=2 ,#if i != self.hparams.n_pools-1 else 4,
                    padding=0,
                    norm=self.hparams.normalization_layer,
                    key=f"{i}_{i+1}_learnedpool",
                )
            )

        voxel_encoder.append(
            VL.Conv1Layer(
                self.hparams.encoder.voxel_encoder[-1],
                self.hparams.S * self.hparams.dim_latent,
                norm=self.hparams.normalization_layer,
                key=f"L_conv"
            )
        )

        self.encoder = nn.ModuleDict({
            "point": PE.LinearEncoder(
                dim_in=self.hparams.data.input_dim + self.hparams.data.n_features,
                encoder=self.hparams.encoder.point_encoder,
                norm=self.hparams.normalization_layer
            ),
            "voxel": nn.Sequential(*voxel_encoder)
        })

        self.activated_transformations = OrderedDict({
            decoder: True for decoder in self.hparams.transformations
        })

        self.decoders = nn.ModuleDict({
            decoder: PD.LinearDecoder(
                dim_in=self.hparams.dim_latent,
                decoder=self.hparams.decoders.decoders,
                dim_out=self.DECODERS_N_PARAMS[decoder] * (self.hparams.K if decoder.split("_")[-1] == "LK" else 1),
                norm=self.hparams.normalization_layer,
                end_with_bias=True
            ) for decoder in self.hparams.transformations
        })

    def on_train_start(self):
        results = {
            "Loss/val": float("nan"),
            "Loss/train": float("nan"),
            "Losses/chamfer/val": float("nan"),
            "Losses/chamfer/train": float("nan"),
        }

        self.logger.log_hyperparams(self.hparams, results)

        self.logger.experiment.add_text(
            "model", self.__repr__().replace("\n", "  \n"), global_step=0)

    def do_step(self, batch, batch_idx, tag):

        with torch.no_grad():
            batch_size = batch.batch.max().item() + 1

        out = self.forward(batch, tag, batch_size=batch_size, batch_idx=batch_idx)

        out["batch_size"] = batch_size

        with torch.profiler.record_function(f"LOSS"):
            out["loss"] = self.compute_loss(tag, batch_size, out)

        with torch.no_grad():
            with torch.profiler.record_function(f"LOGGERS"):

                self.log(f'Loss/{tag}', out["loss"], on_step=False,
                         on_epoch=True, batch_size=out["batch_size"])
                         
                if "chamfer" in out.keys() and out["chamfer"] is not None:
                    self.log(
                        f'Losses/chamfer/{tag}', out["chamfer"].mean(), on_step=False, on_epoch=True, batch_size=out["batch_size"])
                
                if "chamfer4D" in out.keys() and out["chamfer4D"] is not None:
                    self.log(
                        f'Losses/chamfer4D/{tag}', out["chamfer4D"].mean(), on_step=False, on_epoch=True, batch_size=out["batch_size"])

                for l in ["l_XP", "l_PX", "l_gamma", "l_gamma0", "l_KL_K", "l_KL_L", "l_Lmean", "l_Kmean", "l_entropie_K", "l_entropie_L", "l_freq_K", "l_freq_L", "l_xytranslate"]:
                    if l in out.keys():
                        self.log(
                            f'Losses/{l}/{tag}', out[l].mean(), on_step=False, on_epoch=True, batch_size=out["batch_size"])

                self.greedy_step(batch, out, batch_idx, tag, out["batch_size"])

        return {k: v if ((k == "loss") or (not isinstance(v, torch.Tensor))) else v.detach() for k, v in out.items()}

    @torch.profiler.record_function(f"FORWARD")
    def forward(self, batch, tag, batch_size, batch_idx):
        if not hasattr(batch, "features"):
            batch.features = self.compute_features(batch)
        encoded, proto_slab, input_slab = self.encode_batch(batch, batch_size, tag)

        protos, out = self.get_transformed_protos(encoded)
        protos = self.shift_protos_to_voxel_grid(protos, proto_slab)

        bkg = None

        self.compute_choice(protos, encoded, batch, proto_slab, batch_size, out)

        if tag != "train" or batch_idx == 0:
            self.do_reconstruction(batch_size, proto_slab, protos, out, bkg)

        self.compute_reconstruction_loss(tag, batch, batch_size, out, protos, proto_slab, None, batch_idx=batch_idx)

        return out

    @torch.no_grad()
    def forward_light(self, batch, tag, batch_size, batch_idx):
        if not hasattr(batch, "features"):
            batch.features = self.compute_features(batch)
        encoded, proto_slab, input_slab = self.encode_batch(batch, batch_size, tag)

        protos, out = self.get_transformed_protos(encoded)
        protos = self.shift_protos_to_voxel_grid(protos, proto_slab)

        bkg = None

        self.compute_choice(protos, encoded, batch, proto_slab, batch_size, out)

        if tag != "train" or batch_idx == 0:
            self.do_reconstruction(batch_size, proto_slab, protos, out, bkg)

        with torch.no_grad():
            y_pred = []
            inst_pred = []

            y_lengths = torch.tensor([rec.size(0) for rec in out["recs"]], device=batch.pos.device)

            max_y_lenght = y_lengths.max()
            y = torch.stack([F.pad(rec, (0, 0, 0, max_y_lenght - rec.size(0)), "constant", 0) for rec in out["recs"]])

            cham_x, cham_y, idx, _ = chamfer_distance(batch.pos_padded[..., :3], y, batch.pos_lenght, y_lengths)

            out["chamfer"] = .5 * (cham_x.sum(-1) / batch.pos_lenght + cham_y.sum(-1) / y_lengths)
            #out["chamfer"] = .5 * (cham_x.sum(-1) + cham_y.sum(-1))

            inst_pred_padded = torch.div(idx, self.hparams.protos.points).long()      

            for item in range(batch_size):#, x_length in enumerate(batch.pos_lenght):
                with torch.no_grad():
                    inst_pred.append(inst_pred_padded[item, :batch.pos_lenght[item]])
                    choice_item = out["choice"][proto_slab[:, 0] == item].flatten()
                    choice_item = choice_item[choice_item != -1]

                    if tag == "test":
                        choice_item = torch.arange(self.hparams.protos.points, device=choice_item.device).unsqueeze(-1) + self.hparams.protos.points * choice_item.unsqueeze(0)
                        y_pred.append(choice_item.T.flatten()[idx[item, :batch.pos_lenght[item]]])
                    else:
                        y_pred.append(choice_item[inst_pred[-1]])

            out["y_pred"] = torch.cat(y_pred)
            out["inst_pred"] = torch.cat(inst_pred)

        return out

    @torch.profiler.record_function(f"DO_RECONSTRUCTION")
    def do_reconstruction(self, batch_size, proto_slab, protos, out, bkg):
        recs = [torch.zeros((0, 3), device=protos.device, dtype=protos.dtype) for _ in range(batch_size)]
        recs_k = [torch.zeros((0, ), device=protos.device, dtype=out["choice"].dtype) for _ in range(batch_size)]
        for s in range(proto_slab.size(0)):
            recs[proto_slab[s, 0]] = torch.cat([recs[proto_slab[s, 0]]] + [
                protos[s, l, k] for l, k in enumerate(out["choice"][s]) if k != -1
            ], 0)
            recs_k[proto_slab[s, 0]] = torch.cat([recs_k[proto_slab[s, 0]]] + [
                torch.tensor([k], device=recs_k[proto_slab[s, 0]].device, dtype=recs_k[proto_slab[s, 0]].dtype) for l, k in enumerate(out["choice"][s]) if k != -1
            ], 0)

        out["recs"] = recs
        out["recs_k"] = recs_k

    @torch.profiler.record_function(f"LOSS_CHAMFER")
    def compute_reconstruction_loss(self, tag, batch, batch_size, out, protos, proto_slab, bkg=None, batch_idx=None):
        if "recs" not in out.keys():
            out["chamfer"] = None
            out["recs"] = None
        else:
            y_pred = []
            inst_pred = []

            y_lengths = torch.tensor([rec.size(0) for rec in out["recs"]], device=batch.pos.device)

            max_y_lenght = y_lengths.max()
            y = torch.stack([F.pad(rec, (0, 0, 0, max_y_lenght - rec.size(0)), "constant", -1.) for rec in out["recs"]])

            cham_x, cham_y, idx, _ = chamfer_distance(batch.pos_padded[..., :3], y, batch.pos_lenght, y_lengths)

            mask = torch.logical_and(
                y[..., :2] >= 0,
                y[..., :2] <= self.hparams.data.max_xy
            ).all(-1)

            out["chamfer"] = .5 * (cham_x.sum(-1) / batch.pos_lenght + (mask * cham_y).sum(-1) / mask.sum(-1))


            ppoints = self.hparams.protos.points
            if self.hparams.protos.name == "cube_diff":
                ppoints = 6*6*6
            if self.hparams.protos.name == "superquadrics_diff":
                ppoints = 162

            if tag == "test" and hasattr(self, "_protosfeat"):
                assert y.size(0) == 1

                choice = out["choice"][0][out["choice"][0] != -1]
                feat = self.get_protosfeat().squeeze()[choice]
                feat = feat.unsqueeze(0).unsqueeze(-1).repeat_interleave(ppoints, 1)

                cham_x, cham_y, idx, _ = chamfer_distance(batch.pos_padded*self.lambda_xyz_feat / self.lambda_xyz_feat[0], torch.cat([y, feat], -1)*self.lambda_xyz_feat / self.lambda_xyz_feat[0], batch.pos_lenght, y_lengths)

                out["chamfer4D"] = .5 * (cham_x.sum(-1) / batch.pos_lenght + (mask * cham_y).sum(-1) / mask.sum(-1))

            inst_pred_padded = torch.div(idx, ppoints).long()  

            for item in range(batch_size):
                with torch.no_grad():
                    inst_pred.append(inst_pred_padded[item, :batch.pos_lenght[item]])
                    choice_item = out["choice"][proto_slab[:, 0] == item].flatten()
                    choice_item = choice_item[choice_item != -1]

                    if tag == "test":
                        choice_item = torch.arange(ppoints, device=choice_item.device).unsqueeze(-1) + ppoints * choice_item.unsqueeze(0)
                        y_pred.append(choice_item.T.flatten()[idx[item, :batch.pos_lenght[item]]])
                    else:
                        y_pred.append(choice_item[inst_pred[-1]])

            out["y_pred"] = torch.cat(y_pred)
            out["inst_pred"] = torch.cat(inst_pred)

    def compute_choice(self, protos, encoded, batch, proto_slab, batch_size, out):
        choice = torch.randint(self.hparams.K, (protos.size(0), self.hparams.S), dtype=torch.long, device=protos.device)
        
        out["choice"] = choice

    @torch.profiler.record_function(f"ENCODE")
    def encode_batch(self, batch, batch_size, tag):

        with torch.no_grad():
            highres_ind = torch.floor(batch.pos / self.voxelizer).to(torch.int32)
            highres_ind = torch.minimum(highres_ind, self.N_voxels - 1).clamp(min=0)
            centered_pos = 2 * (batch.pos / self.voxelizer - (highres_ind + .5))
            features = torch.cat([batch.features, centered_pos], -1)
            highres_ind = torch.cat([batch.batch.to(torch.int32).unsqueeze(-1), highres_ind], -1)

            lowres_ind, highres2lowres = torch.unique(highres_ind, dim=0, return_inverse=True)
        
        features = self.encoder["point"](features)


        #print(features.size(), torch.isnan(features).any(), torch.isinf(features).any())

        features = torch_scatter.scatter_max(features, highres2lowres, 0)[0]

        spconvtensor = spconv.SparseConvTensor(
            features,
            lowres_ind,
            self.N_voxels_tuple,
            batch_size
        )
        spconvtensor = self.encoder["voxel"](spconvtensor)

        #print(spconvtensor.features.size(), torch.isnan(spconvtensor.features).any(), torch.isinf(spconvtensor.features).any())

        #print(spconvtensor.indices)

        input_slab = highres_ind[..., :-1]
        input_slab[..., 1:] = torch.div(input_slab[..., 1:], self.thin2coarse, rounding_mode="floor")

        out_features = spconvtensor.features.view(-1, self.hparams.S, self.hparams.dim_latent)
        return out_features, spconvtensor.indices, input_slab

    def compute_loss(self, tag, batch_size, out):
        return out["chamfer"].mean()

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, 'test')

    def log_step(self):
        self.log(f'step', 1. + self.current_epoch,
                 on_step=False, on_epoch=True, batch_size=1)

    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)

        self.log_step()

    def on_validation_epoch_end(self, *args, **kwargs):
        super().on_validation_epoch_end(*args, **kwargs)

        self.log_step()

    def on_test_epoch_end(self, *args, **kwargs):
        super().on_test_epoch_end(*args, **kwargs)

        self.log_step()

    def configure_optimizers(self):
        parameters = [
            {"params": [], "name": "base"},
            {"params": self.encoder.parameters(), "name": "encoder"}            
        ]
        
        if hasattr(self, "_protos") and self.hparams.protos.name not in ["cube_diff"]:
            parameters.append({
                "params": self._protos if self.hparams.protos.name in ["points", "superquadrics_diff"] else self._protos.parameters(), "name": "protos",
                'weight_decay': self.hparams.protos.optim.weight_decay,
                'lr': self.hparams.protos.optim.lr,
                'eps': self.hparams.protos.optim.eps
            })

        if hasattr(self, "_protosfeat"):
            parameters.append({
                "params": self._protosfeat,
                "name": "protosfeat", 'weight_decay': 0
            })

        if self.hparams.protos.learn_specific_scale:
            parameters.append({
                "params": self._protosscale,
                "name": "protosscale", 'weight_decay': 0
            })
        
        for decoder in self.decoders:
            parameters.append({
                "params": self.decoders[decoder].parameters(),
                "name": decoder
            })
            
        if hasattr(self, "chooser"):
            parameters.append({
                "params": self.chooser.parameters(),
                "name": "chooser"
            })
            

        optimizer = getattr(torch.optim, self.hparams.optim._target_)(
            parameters,
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.weight_decay,
            #eps=self.hparams.optim.eps,
            #momentum=self.hparams.optim.momentum
        )

        return {
            "optimizer": optimizer,
        }
