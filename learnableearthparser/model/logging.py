import torch

import numpy as np

class LoggingModel:

    def do_greedy_step(self):
        return (self.current_epoch % int(self.hparams.log_every_n_epochs)) == 0

    @torch.no_grad()
    def greedy_step(self, batch, out, batch_idx, tag, batch_size):

        if tag == "train":
            n = (out["choice"] != -1).sum()
            self.log(f'Prediction/N_protos', n / batch_size, on_step=False, on_epoch=True, batch_size = batch_size)

            if hasattr(self, "_protosfeat"):
                for i, val in enumerate(self.get_protosfeat().flatten()):
                    self.log(f'_protosfeat/{i}', val.item(), on_step=False, on_epoch=True, batch_size = batch_size)

            if hasattr(self, "_protosscale") and self.hparams.protos.learn_specific_scale:
                for k in range(self.hparams.K):
                    for xzy, val in zip(["x", "y", "z"], self.get_protosscale()[k].flatten()):
                        self.log(f'_protosscale/{xzy}_{k}', val.item(), on_step=False, on_epoch=True, batch_size = batch_size)


        if batch_idx == 0 and self.do_greedy_step():
            with torch.no_grad():
                if tag == "train":
                    self.greedy_model()
                    self.greedy_histograms(batch, out)
                if out["recs"] is not None:
                    self.greedy_pcs(batch, out["recs"], tag)

    @torch.no_grad()
    def greedy_model(self):
        # Plot Protos
        points, faces = None, None

        config_dict = {
            'material': {
                'cls': 'PointsMaterial',
                'size': 0.05
            }
        }

        protos = self.get_protos(points)

        for ixyz, xyz in enumerate(["x", "y", "z"]):
            self.logger.experiment.add_histogram(
                    f"protos/{xyz}",
                    protos[..., ixyz].detach().cpu().flatten(),
                    global_step=self.current_epoch
                )       

        mini, maxi = protos.min(1)[0].unsqueeze(1), protos.max(1)[0].unsqueeze(1)
        
        colors = (255 * (protos - mini) / (maxi - mini + 10e-8)).int()

        self.logger.experiment.add_mesh(
            f"protos_pointcloud", protos,
            colors=colors, faces=faces,
            config_dict=config_dict, global_step=self.current_epoch
        )

    
    @torch.no_grad()    
    def greedy_histograms(self, batch, out):

        if "kappa_presoftmax" in out.keys():
            self.logger.experiment.add_histogram(
                f"kappa/presoftmax",
                out["kappa_presoftmax"].detach().cpu().flatten(),
                global_step=self.current_epoch
            )
        if "kappa_postsoftmax" in out.keys():
            self.logger.experiment.add_histogram(
                f"kappa/postsoftmax",
                out["kappa_postsoftmax"].detach().cpu().flatten(),
                global_step=self.current_epoch
            )

        if hasattr(self, "_protos"):
            for exp in ["exp_avg_sq", "exp_avg"]:
                if exp in self.optimizers().state[self._protos].keys():
                    self.logger.experiment.add_histogram(
                        f"protos_optim/{exp}",
                        self.optimizers().state[self._protos][exp].detach().cpu().flatten(),
                        global_step=self.current_epoch
                    )

        for i, (feat, featname) in enumerate(zip(batch.features.T, self.trainer.datamodule.get_feature_names())):  
            self.logger.experiment.add_histogram(
                    f"features/{featname}",
                    feat.detach().cpu().flatten(),
                    global_step=self.current_epoch
                )

        for ixyz, xyz in enumerate(["x", "y", "z"]):
            self.logger.experiment.add_histogram(
                    f"positions/{xyz}",
                    batch.pos[:, ixyz].detach().cpu().flatten(),
                    global_step=self.current_epoch
                )

        for o in self.activated_transformations.keys():
            if o in out.keys() and out[o] is not None:
                if "translation" in o:
                    www = out[o].squeeze()
                    mini = www.min(0)[0]
                    maxi = www.max(0)[0]

                    for ixyz, xyz in enumerate(["x", "y", "z"]):
                        self.logger.experiment.add_histogram(
                            f"{o}/spatial_extend/{xyz}",
                            (maxi - mini)[..., ixyz].detach().cpu().flatten(),
                            global_step=self.current_epoch
                        )

                        if hasattr(self.decoders[o][-1], "bias") and self.decoders[o][-1].bias is not None:
                            self.logger.experiment.add_histogram(
                                f"{o}/ll_bias/{xyz}",
                                self.decoders[o][-1].bias[ixyz::3].detach().cpu().flatten(),
                                global_step=self.current_epoch
                            )

                        self.logger.experiment.add_histogram(
                            f"{o}/ll_weight/{xyz}",
                            self.decoders[o][-1].weight[ixyz::3].detach().cpu().flatten(),
                            global_step=self.current_epoch
                        )

                    
                    
                if out[o][0, 0, 0].numel() == 3:
                    for ixyz, xyz in enumerate(["x", "y", "z"]):
                        self.logger.experiment.add_histogram(
                            f"{o}/{xyz}",
                            out[o][..., ixyz].detach().cpu().flatten(),
                            global_step=self.current_epoch
                        )
                else:
                    self.logger.experiment.add_histogram(
                        f"{o}",
                        out[o].detach().cpu().flatten(),
                        global_step=self.current_epoch
                    )

    @torch.no_grad()        
    def greedy_pcs(self, batch, recs, tag):

        self.log(f'NpointsX/{tag}', batch.pos_lenght.float().mean().item(), on_step=False,
                on_epoch=True, batch_size=1)



        pos, rec = batch.pos[batch.batch == 0], recs[0]

        NMAX = 2**13 if pos.size(0) > 2**13 else pos.size(0)
        if pos.size(0) > NMAX:
            pos = pos[np.random.choice(pos.size(0), NMAX, replace=True)]
        rec = rec[np.random.choice(rec.size(0), NMAX, replace=True)]
            
        pc = torch.cat([pos.unsqueeze(0), rec.unsqueeze(0)], 0)
        pc = pc - pc.view(-1, 3).mean(0)


        mini, maxi = pc.min(1)[0].unsqueeze(1), pc.max(1)[0].unsqueeze(1)
        colors = (255 * (pc - mini) / (maxi - mini + 10e-8)).int()

        config_dict = {
            'material': {
                'cls': 'PointsMaterial',
                'size': 0.05 * (1 + 3*("lidar" in self.hparams.data.name))
            },
        }

        colors[1] *= 0
        self.logger.experiment.add_mesh(
            f"pred_{tag}", pc.view(1, -1, 3), colors=colors.view(1, -1, 3),
            faces=None, config_dict=config_dict, global_step=self.current_epoch
        )