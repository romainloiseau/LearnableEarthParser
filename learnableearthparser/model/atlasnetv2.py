import torch

from .base import BaseModel

from ..utils import chamfer_distance


class AtlasNetV2(BaseModel):

    @torch.profiler.record_function(f"CHOICE")
    def compute_choice(self, protos, encoded, batch, proto_slab, batch_size, out):
        out["choice"] = torch.arange(self.hparams.K, dtype=torch.long, device=protos.device).unsqueeze(0).repeat(protos.size(0), 1)

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


        l_XP = cham_x.view(batch_size, self.hparams.S, self.hparams.K, -1)

        l_XP = torch.diagonal(l_XP.permute(0, 3, 1, 2), dim1=-2, dim2=-1)
        
        l_XP = l_XP.min(-1)[0]

        
        out["l_PX"] = torch.diagonal(cham_y.mean(-1).view(-1, self.hparams.S, self.hparams.K), dim1=-2, dim2=-1).mean(-1)
        out["l_XP"] = l_XP.sum(-1) / batch.pos_lenght.float()


        if self.hparams.distance != "xyz":
            protos = protos[..., :3]

        with torch.no_grad():
            super().compute_reconstruction_loss(tag, batch, batch_size, out, protos, proto_slab, bkg, batch_idx)
        
    def compute_loss(self, tag, batch_size, out):
        return (
            self.hparams.lambda_XP * out["l_XP"]
            + self.hparams.lambda_PX * out["l_PX"]
        ).mean()