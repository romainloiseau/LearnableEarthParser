import torch
import torch.nn.functional as F
from .point import decoders as PD

from .base import BaseModel

from ..utils import chamfer_distance

@torch.jit.script
def compute_freq_loss(choice_K, choice_L, epsilon_freq_K: float, epsilon_freq_L: float, K: int, S: int):
    lK = choice_K.sum(0).sum(0)
    lK = lK / lK.sum()
    lK = - lK.clamp(max=epsilon_freq_K / K) / epsilon_freq_K

    lL = choice_L.sum(0)
    lL = lL / lL.sum()
    lL = - lL.clamp(max=epsilon_freq_L / S) / epsilon_freq_L

    return 1 + lK.sum(), 1 + lL.sum()

@torch.jit.script
def compute_l_XP(kappa_presoftmax: torch.Tensor, choice_L: torch.Tensor, cham_x: torch.Tensor, x_lengths_LK: torch.Tensor, S: int, K: int) -> torch.Tensor:
    for_wsum = kappa_presoftmax[..., 1:]
    for_wsum = torch.exp(for_wsum.unsqueeze(-2) - for_wsum.unsqueeze(-1)).sum(-1)
    cham_x = (cham_x.view(-1, S, K, cham_x.size(-1)) / for_wsum.unsqueeze(-1)).sum(-2)
    x_lengths_L = x_lengths_LK[..., 0]

    sorted_cham_x, indices = torch.sort(cham_x, 1, descending=False)
    epsilon_ordered = torch.gather(choice_L.unsqueeze(-1).repeat(1, 1, indices.size(-1)), 1, indices) #torch.equal(torch.gather(cham_x, 1, indices),sorted_cham_x) is True
    cumprod = torch.cumprod(1 - epsilon_ordered, dim=1)
    cumprod = torch.cat([torch.ones((cumprod.size(0), 1, cumprod.size(-1)), device=cumprod.device, dtype=cumprod.dtype), cumprod[:, :-1]], 1)
    l_XP = sorted_cham_x * epsilon_ordered * cumprod
    l_XP = (l_XP.sum(-1) / x_lengths_L).sum(-1) #l_XP.sum(-1).sum(-1) #(l_XP.sum(-1) / x_lengths).sum(-1)  ---> fixed chamfer loss

    return l_XP

@torch.jit.script
def compute_l_PX(y: torch.Tensor, choice_K: torch.Tensor, cham_y: torch.Tensor, max_xy: float, S: int, K: int) -> torch.Tensor:
    with torch.no_grad():
        mask = torch.logical_and(
            y[..., :2] >= 0,
            y[..., :2] <= max_xy
        ).all(-1)
        mask_sum = mask.sum(-1)
        mask_sum_is_zero = mask_sum==0
        mask_sum[mask_sum_is_zero] = 1
    cham_y = (cham_y * mask).sum(-1) / mask_sum
    cham_y[mask_sum_is_zero] = 1.
    l_PX = choice_K.flatten() * cham_y
    l_PX = l_PX.view(-1, S*K).sum(-1)
    return l_PX / S

@torch.jit.script
def compute_translate_loss(translation_L: torch.Tensor) -> torch.Tensor:
    xytranslate = F.softshrink(translation_L[..., :2], 1.)**2

    return xytranslate.mean(0).sum()

@torch.jit.script
def compute_gamma_loss(choice_L):
    return choice_L.sum(-1), torch.clamp(1 - choice_L.sum(-1), min=0)


class OursModel(BaseModel):

    def __initnets__(self):
        super().__initnets__()
        
        self.chooser = PD.LinearDecoder(
            dim_in = self.hparams.dim_latent,
            decoder = self.hparams.decoders.decoders, 
            dim_out = self.hparams.K + 1,
            norm = self.hparams.normalization_layer, end_with_bias=False
        )
        self.register_buffer("normalize_proba", 1. / torch.tensor(self.hparams.decoders.decoders[-1])**.5, persistent=False)

        if self.hparams.name == "superquadrics":
            self.SUPERQUADRIC_MODE = "train"

    def compute_logits(self, scene_features):
        return self.chooser(scene_features) * self.normalize_proba

    @torch.profiler.record_function(f"CHOICE")
    def compute_choice(self, protos, encoded, batch, proto_slab, batch_size, out):
        kappa = self.compute_logits(encoded.flatten(0, 1)).view(-1, self.hparams.S, self.hparams.K + 1)
        out["kappa_presoftmax"] = kappa
        
        if hasattr(self, "remove_proto"):
            to_share = kappa[:, :, self.remove_proto].sum(-1, keepdim=True)
            kappa[:, :, 1:] = kappa[:, :, 1:] + to_share / (kappa.size(-1) - 1)
            kappa[:, :, self.remove_proto] = kappa.min()
        
        kappa = torch.nn.functional.softmax(kappa, -1)
        out["kappa_postsoftmax"] = kappa

        choice_L = 1 - kappa[:, :, 0]
        choice_K = kappa[:, :, 1:]
        
        with torch.no_grad():
            if hasattr(self, "remove_proto"):
                choice = kappa.argmax(-1)
            else:
                choice = torch.multinomial(kappa.flatten(0, 1), 1).view(-1, choice_L.size(1))
            zero = choice.sum(-1) == 0
            most_probable_slot = kappa[zero, :, 0].min(-1)[1]
            choice[zero, most_probable_slot] = 1 + kappa[zero, most_probable_slot, 1:].max(-1)[1]

            choice = choice - 1
            
        out["choice"] = choice
        out["choice_L"] = choice_L
        out["choice_K"] = choice_K

    @torch.profiler.record_function(f"LOSS_SUPERQUADRIC")
    def compute_reconstruction_loss(self, tag, batch, batch_size, out, protos, proto_slab, bkg=None, batch_idx=None):

        assert proto_slab.size(0) == batch_size, f"{proto_slab.size(0)} != {batch_size}"

        if self.hparams.distance != "xyz":
            protos = torch.cat([
                protos,
                self.get_protosfeat().unsqueeze(0).unsqueeze(0).unsqueeze(-2).repeat((protos.size(0), protos.size(1), 1, protos.size(-2),1))
            ], -1)

        x_lengths_LK = batch.pos_lenght.unsqueeze(-1).unsqueeze(-1).repeat(1, self.hparams.S, self.hparams.K)
        y = protos.flatten(0, 2)
        
        x_LK = batch.pos_padded.unsqueeze(1).repeat(1, self.hparams.S*self.hparams.K, 1, 1).flatten(0, 1)
        cham_x, cham_y, _, _ = chamfer_distance(x_LK*self.lambda_xyz_feat, y*self.lambda_xyz_feat, x_lengths_LK.flatten(), y_lengths = None)
        
        out["l_PX"] = compute_l_PX(y, out["choice_K"], cham_y, self.hparams.data.max_xy, self.hparams.S, self.hparams.K)
        out["l_XP"] = compute_l_XP(out["kappa_presoftmax"], out["choice_L"], cham_x, x_lengths_LK, self.hparams.S, self.hparams.K)

        out["l_gamma"], out["l_gamma0"] = compute_gamma_loss(out["choice_L"])

        out["l_freq_K"], out["l_freq_L"] = compute_freq_loss(
            out["choice_K"], out["choice_L"],
            self.hparams.epsilon_freq_K, self.hparams.epsilon_freq_L,
            self.hparams.K, self.hparams.S
        )

        out["l_xytranslate"] = compute_translate_loss(out["translation_L"])

        if self.hparams.distance != "xyz":
            protos = protos[..., :3]

        with torch.no_grad():
            super().compute_reconstruction_loss(tag, batch, batch_size, out, protos, proto_slab, bkg, batch_idx)
        
    def compute_loss(self, tag, batch_size, out):
        return (
            self.hparams.lambda_XP * out["l_XP"]
            + self.hparams.lambda_PX * out["l_PX"]
            + self.hparams.lambda_gamma * out["l_gamma"]
            + self.hparams.lambda_gamma0 * out["l_gamma0"]
            + self.hparams.lambda_freq_K * out["l_freq_K"]
            + self.hparams.lambda_freq_L * out["l_freq_L"]
            + self.hparams.lambda_xytranslate * out["l_xytranslate"]
        ).mean()

    @torch.no_grad()    
    def greedy_histograms(self, batch, out):
        super().greedy_histograms(batch, out)
        
        with torch.no_grad():
        
            self.logger.experiment.add_histogram(
                f"choice_L",
                out["choice_L"].detach().cpu().flatten(),
                global_step=self.current_epoch
            )

            self.logger.experiment.add_histogram(
                f"choice_K",
                out["choice_K"].detach().cpu().flatten(),
                global_step=self.current_epoch
            )

            self.logger.experiment.add_histogram(
                f"choice_K_max",
                out["choice_K"].max(-1)[0].detach().cpu().flatten(),
                global_step=self.current_epoch
            )

            if hasattr(self, "chooser"):
                self.logger.experiment.add_histogram(
                    f"chooser_LK/weight",
                    self.chooser[-1].weight.detach().cpu().flatten(),
                    global_step=self.current_epoch
                )
                if hasattr(self.chooser[-1], "bias") and self.chooser[-1].bias is not None:
                    self.logger.experiment.add_histogram(
                        f"chooser_LK/bias",
                        self.chooser[-1].bias.detach().cpu().flatten(),
                        global_step=self.current_epoch
                    )