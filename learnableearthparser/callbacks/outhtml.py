import omegaconf
import torch
import matplotlib as mpl
import numpy as np
import os.path as osp
from .base import DTICallback
from datetime import datetime
import torch_scatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
import math
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from ..utils.icosphere import generate_icosphere
from ..utils.labels import apply_learning_map


class OutHTML(DTICallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_val_samples = self.myhparams.n_val_samples
        self.samples_per_rows = self.myhparams.samples_per_rows
        self.points_per_scene = self.myhparams.points_per_scene

        self.margin = 12
        self.scene_size = 400

    def on_train_start(self, trainer, pl_module):
        self.val_samples = [trainer.datamodule.val_dataset[np.random.randint(
            len(trainer.datamodule.val_dataset))] for n in range(self.n_val_samples)]
        return super().on_train_start(trainer, pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_samples = [trainer.datamodule.test_dataset[np.random.randint(
            len(trainer.datamodule.test_dataset))] for n in range(self.n_val_samples)]
        return super().on_test_start(trainer, pl_module)

    def get_header(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        header = ""
        header += "<head>\n"
        header += '\t<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n'
        header += """
            <style>
            table {
            font-family: arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            }
            td, th {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
            }
            tr:nth-child(even) {
            background-color: #dddddd;
            }
            </style>\n
        """
        header += """
            <style>
            * {
            box-sizing: border-box;
            }
            .column1 {
            float: left;
            width: 100%;
            padding: 10px;
            }
            .column2 {
            float: left;
            width: 50%;
            padding: 10px;
            }
            .column3 {
            float: left;
            width: 33%;
            padding: 10px;
            }
            .column4 {
            float: left;
            width: 25%;
            padding: 10px;
            }

            /* Clear floats after the columns */
            .row:after {
            content: "";
            display: table;
            clear: both;
            }

            /* Responsive layout - makes the two columns stack on top of each other instead of next to each other */
            @media screen and (max-width: 800px) {
            .column2 {
                width: 100%;
            }
            .column3 {
                width: 100%;
            }
            .column4 {
                width: 100%;
            }
            }
            </style>
        """
        header += "</head>\n"
        return header

    def add_hparams(self, hparams, data_hparams):
        
        html = """<div class="row"><div class="column3">"""

        html += "<h2>Model</h2><table>\n"
        html = self.add_table_row(html, "name", hparams.name, style="th")
        for k, v in hparams.items():
            if k not in ["data", "callbacks", "name"]:
                html = self.add_table_row(html, k, v)

        html += """</table></div><div class="column3"><h2>Data</h2><table>"""
        html = self.add_table_row(html, "name", data_hparams.name, style="th")
        for k, v in data_hparams.__dict__.items():
            if k != "data_dir":
                html = self.add_table_row(html, k, v)
        html += """</table></div><div class="column3"><h2>Callbacks</h2><table>"""
        for i, callback in enumerate(hparams["callbacks"]):
            html = self.add_table_row(html, "_target_", getattr(hparams["callbacks"], callback)._target_, style="th")
            for k, v in getattr(hparams["callbacks"], callback).items():
                if k != "_target_":
                    html = self.add_table_row(html, k, v)

        html += "</table></div></div>"

        return html

    def add_table_row(self, html, k, v, style="td"):
        html += f"<tr><{style}>{k}</{style}><{style}>"

        if isinstance(v, omegaconf.listconfig.ListConfig):
            html += " - ".join([str(vv) for vv in v])
        else:
            html += f"{v}"

        return html + f"</{style}></tr>"

    def get_title(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        title = ""
        title += self.add_text("div",
                               datetime.now().strftime("%d/%m/%Y - %H:%M:%S"))
        title += self.add_text(
            "h1", f"Training {pl_module.hparams.name} on data {pl_module.hparams.data.name}")

        title += self.add_hparams(pl_module.hparams, trainer.datamodule.myhparams)
        return title

    def add_text(self, block, text):
        return f"\t<{block}>\n\t\t{text}\n\t</{block}>\n"

    @torch.no_grad()
    def get_protos(self, protos):
        
        mini, maxi = np.expand_dims(protos.min(
            1), 1), np.expand_dims(protos.max(1), 1)
        colors = (protos - mini) / (maxi - mini)

        K = protos.shape[0]

        n_cols = min(self.samples_per_rows, K)
        n_rows = 1 + (K - 1)//self.samples_per_rows
        specs = [[{"type": "scene"}
                  for _ in range(n_cols)]for _ in range(n_rows)]

        layout = go.Layout(
            title="Prototypes",
            scene=dict(aspectmode='data', ),
            width=n_cols*self.scene_size,
            height=n_rows*self.scene_size,
            margin=dict(l=self.margin, r=self.margin,
                        b=self.margin, t=4*self.margin),
            uirevision=True
        )
        figure = go.Figure(
            layout=layout
        )
        figure = make_subplots(
            rows=n_rows, cols=n_cols,
            specs=specs,
            shared_xaxes=True,
            shared_yaxes=True,
            figure=figure,
            horizontal_spacing=.01,
            vertical_spacing=.01,
        )

        mini = mini.min(0)[0]
        maxi = maxi.max(0)[0]

        for k in range(K):
            figure.add_trace(
                go.Scatter3d(
                    x=protos[k, :, 0], y=protos[k, :, 1], z=protos[k, :, 2],
                    mode='markers', scene="scene1",
                    marker={"size": 2, "color": colors[k]},
                    legendgroup='points', showlegend=k == 0, name="points"
                ),
                row=1 + k//n_cols, col=1 + k % n_cols
            )
            figure.add_trace(
                go.Scatter3d(
                    name="",
                    visible=True,
                    showlegend=False,
                    opacity=0,
                    hoverinfo='none',
                    x=[mini[0],maxi[0]],
                    y=[mini[1],maxi[1]],
                    z=[mini[2],maxi[2]]
                ),
                row=1 + k//n_cols, col=1 + k % n_cols
            )
        figure.update_layout(scene_aspectmode='data')
        figure.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.,
            xanchor="left",
            x=0
        ))

        return figure.to_html(full_html=False, include_plotlyjs='cdn') + "\n<br><br>\n"

    @torch.no_grad()
    def get_inferences(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        
        mini = [0, 0, 0] #Batch.from_data_list(self.val_samples).pos.min(0)[0]
        maxi = [self.max_xy, self.max_xy, self.max_z] #Batch.from_data_list(self.val_samples).pos.max(0)[0]

        n_cols = min(self.samples_per_rows, self.n_val_samples)
        n_rows = 1 + (self.n_val_samples - 1)//self.samples_per_rows
        specs = [[{"type": "scene"}
                  for _ in range(n_cols)]for _ in range(n_rows)]

        layout = go.Layout(
            title="Inferences",
            scene=dict(aspectmode='data', ),
            width=n_cols*self.scene_size,
            height=n_rows*self.scene_size,
            margin=dict(l=self.margin, r=self.margin,
                        b=self.margin, t=4*self.margin),
            uirevision=True
        )
        figure = go.Figure(
            layout=layout
        )

        figure = make_subplots(
            rows=n_rows, cols=n_cols,
            specs=specs,
            shared_xaxes=True,
            shared_yaxes=True,
            figure=figure,
            horizontal_spacing=.01,
            vertical_spacing=.01,
        )

        is_training = pl_module.training
        pl_module = pl_module.eval()
        for n in range(self.n_val_samples):
            out = pl_module.forward(Batch.from_data_list([self.val_samples[n]]).to(
                pl_module.device), "val", batch_size=1, batch_idx=0)

            color = 0.01 + 0.98*(self.val_samples[n].pos - self.val_samples[n].pos.min(0)[0]) / (
                self.val_samples[n].pos.max(0)[0] - self.val_samples[n].pos.min(0)[0])

            if self.points_per_scene != 0:
                choice = np.random.choice(color.size(
                    0), self.points_per_scene, replace=True)
                pos = self.val_samples[n].pos[choice].to(torch.float16)
                color = (255 * color[choice]).to(torch.uint8)
            else:
                pos = self.val_samples[n].pos.to(torch.float16)
                color = (255 * color).to(torch.uint8)
            figure.add_trace(
                go.Scatter3d(
                    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                    mode='markers', scene="scene1",
                    marker={"size": 2, "color": color.detach().cpu().numpy()},
                    legendgroup='input', showlegend=n == 0, name="input"
                ),
                row=1 + n//n_cols, col=1 + n % n_cols
            )

            color = np.concatenate([pl_module.hparams.protos.points*[c.item()] for c in out["choice"].flatten() if c != -1])
            
            out_recs = out["recs"][0].detach().cpu().numpy()

            assert out_recs.shape[0] == color.shape[0]
            
            color_rgb = mpl.colormaps["tab10"](color / (pl_module.hparams.K))
            color_rgb[color == -1, :3] = 0

            if self.points_per_scene != 0:
                choice_recs = np.random.choice(
                    out_recs.shape[0], self.points_per_scene, replace=True)
                out_recs = out_recs[choice_recs]
                color_rgb = color_rgb[choice_recs]

            figure.add_trace(
                go.Scatter3d(
                    x=out_recs[:, 0], y=out_recs[:, 1], z=out_recs[:, 2],
                    mode='markers', scene="scene1",
                    marker={"size": 2, "color": color_rgb},
                    legendgroup='output', showlegend=n == 0, name="output"
                ),
                row=1 + n//n_cols, col=1 + n % n_cols
            )
            figure.add_trace(
                go.Scatter3d(
                    name="",
                    visible=True,
                    showlegend=False,
                    opacity=0,
                    hoverinfo='none',
                    x=[mini[0],maxi[0]],
                    y=[mini[1],maxi[1]],
                    z=[mini[2],maxi[2]]
                ),
                row=1 + n//n_cols, col=1 + n % n_cols
            )
            
        figure.update_layout(scene_aspectmode='data')
        figure.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.,
            xanchor="left",
            x=0
        ))

        if is_training:
            pl_module = pl_module.train()
        return figure.to_html(full_html=False, include_plotlyjs='cdn') + "\n<br><br>\n"

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        metrics = trainer.logged_metrics

        columns = {
            "train": [],
            "val": [],
            "test": [],
            "other": []
        }
        for k in metrics.keys():
            if "train" in k:
                columns["train"].append(k)
            elif "val" in k:
                columns["val"].append(k)
            elif "test" in k:
                columns["test"].append(k)
            else:
                columns["other"].append(k)
        
        n_columns = sum([1 if len(v)!=0 else 0 for _, v in columns.items()])

        html = f"""<div class="row">"""
        for k, v in columns.items():
            if len(v) != 0:
                html += f"""<div class=column{n_columns}><h3>{k}</h3><table>"""
                for metric in v:
                    html = self.add_table_row(html, metric, metrics[metric])
                html += "</table></div>"
        html += "</div>"

        return html

    @torch.no_grad()
    def get_test_scene_inference(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        idx = 0
        if "field" in trainer.datamodule.test_dataset.options.name:
            idx = 1
        elif "snsem" in trainer.datamodule.test_dataset.options.name:
            idx = 2

        if "fit" in pl_module.hparams.load_weights:
            modelname = pl_module.hparams.load_weights.split("fit/")[-1].split("/")[0]
        else:
            modelname = pl_module.hparams.load_weights.split("outputs/")[-1].split("/")[0]

        isearthparserdataset = trainer.datamodule.test_dataset.options.max_xy > 5 #"earthparserdataset" in trainer.datamodule.test_dataset.options.name
        issuperquadrics = pl_module.hparams.name == "superquadrics"
        if issuperquadrics:
            pl_module.SUPERQUADRIC_MODE = "plot"
            training_Kpoints = copy.deepcopy(pl_module.hparams.protos.points)
            pl_module.hparams.protos.points = [12, 42, 162, 642, 2562, 10242][2]

        if isearthparserdataset:
            slice_test = np.concatenate([np.array([0]), np.cumsum(trainer.datamodule.test_dataset.items_per_epoch)])

            items = [trainer.datamodule.test_dataset[i] for i in range(slice_test[idx], slice_test[idx + 1])]

            pad = max([i.pos_padded.size(1) for i in items])
            for item in items:
                item.pos_padded = F.pad(item.pos_padded[0], (0, 0, 0, pad - item.pos_padded[0].size(0)), mode="constant", value=0).unsqueeze(0)
        else:
            items = [trainer.datamodule.test_dataset[idx]]

        batch = Batch.from_data_list(items)
        for k in batch.keys:
            if type(batch[k]) is torch.Tensor:
                batch[k] = batch[k].to(pl_module.device)

        out = pl_module.forward_light(batch, batch_size=len(items), batch_idx=0, tag="test")

        delta_xy = 0.25

        for k in batch.keys:
            if type(batch[k]) is torch.Tensor:
                batch[k] = batch[k].cpu()
                
        if isearthparserdataset:
            batch.pos[:, :2] += trainer.datamodule.test_dataset.tiles_unique_selection[idx][batch.batch] * trainer.datamodule.test_dataset.options.max_xy
            batch.pos[:, -1] += trainer.datamodule.test_dataset.tiles_min_z[idx][batch.batch]

        batch.point_y_pred = out["y_pred"].detach().cpu()
        batch.point_inst_pred = out["inst_pred"].detach().cpu()


        if hasattr(trainer.callbacks[0], "best_assign") and pl_module.hparams.name != "superquadrics":
            batch.point_y_pred = torch.from_numpy(trainer.callbacks[0].best_assign)[batch.point_y_pred]

        color="y_pred;inst_pred;y;xyz;rgb;i"

        if isearthparserdataset:
            voxelize = 2. * trainer.datamodule.myhparams.pre_transform_grid_sample
        else:
            voxelize = 0
        ps=5
        max_points = 249000

        if isearthparserdataset:
            batch.point_y = apply_learning_map(batch.point_y, trainer.datamodule.myhparams.learning_map_inv)
            batch.point_y_pred = apply_learning_map(batch.point_y_pred, trainer.datamodule.myhparams.learning_map_inv)
        batch.proto_y_pred = copy.deepcopy(out["y_pred"].detach().cpu() // pl_module.hparams.protos.points)
        torch.save(batch.detach().cpu(), f"{pl_module.hparams.data.name}_{modelname}_input.pt")
        del batch.proto_y_pred
        if isearthparserdataset:
            batch.point_y = apply_learning_map(batch.point_y, trainer.datamodule.myhparams.learning_map)
            batch.point_y_pred = apply_learning_map(batch.point_y_pred, trainer.datamodule.myhparams.learning_map)

            batch.pos[:, :2] += trainer.datamodule.test_dataset.tiles_unique_selection[idx][batch.batch] * delta_xy * trainer.datamodule.test_dataset.options.max_xy

        if voxelize:
            choice = torch.unique((batch.pos / voxelize).int(), return_inverse=True, dim=0)[1]
            for attr in ["point_y", "point_y_pred", "point_inst", "point_inst_pred"]:
                if hasattr(batch, attr):
                    setattr(batch, attr, torch_scatter.scatter_sum(F.one_hot(getattr(batch, attr).squeeze().long()), choice, 0).argmax(-1))
            for attr in ["pos", "features"]:
                if hasattr(batch, attr):
                    setattr(batch, attr, torch_scatter.scatter_mean(getattr(batch, attr), choice, 0))
            for attr in ["intensity", "rgbi", "rgb"]:
                if hasattr(batch, attr):
                    setattr(batch, attr, torch_scatter.scatter_max(getattr(batch, attr), choice, 0)[0])
        if batch.pos.size(0) > max_points:
            keep = torch.randperm(batch.pos.size(0))[:max_points]
            for attr in ["point_y", "point_y_pred", "point_inst", "point_inst_pred", "pos", "features", "intensity", "rgbi", "rgb"]:
                if hasattr(batch, attr):
                    setattr(batch, attr, getattr(batch, attr)[keep])

        datadtype = torch.float16
        margin = int(0.02 * 600)
        layout = go.Layout(
            title="Scene prediction",
            width=1000,
            height=600,
            margin=dict(l=margin, r=margin, b=margin, t=4*margin),
            uirevision=True,
        )
        fig = go.Figure(
            layout=layout
        )
        fig.add_trace(go.Scatter3d(
                x=batch.pos[:, 0].to(datadtype), y=batch.pos[:, 1].to(datadtype), z=batch.pos[:, 2].to(datadtype),
                mode='markers',
                marker=dict(size=ps, color=trainer.datamodule.get_color_from_item(batch, color.split(";")[0])),
                name="input"
            )
        )
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{"marker": dict(size=ps, color=trainer.datamodule.get_color_from_item(batch, c))}, [0]],
                        label=trainer.datamodule.get_label_from_raw_feature(c),
                        method="restyle"
                    ) for c in color.split(";")
                ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.05,
            xanchor="left",
            y=0.88,
            yanchor="top"
            ),
        ]
        fig.update_layout(updatemenus=updatemenus)


        fig.update_layout(
            scene_aspectmode='data',
        )

        out_string = fig.to_html(full_html=False, include_plotlyjs='cdn') + "\n<br><br>\n"
        del fig
        
        layout2 = go.Layout(
            title="Scene reconstruction",
            width=1000,
            height=600,
            margin=dict(l=margin, r=margin, b=margin, t=4*margin),
            uirevision=True,
        )
        fig2 = go.Figure(
            layout=layout2
        )

        if isearthparserdataset:
            reconstruction = torch.cat([
                out["recs"][i].view(-1, pl_module.hparams.protos.points, 3).detach() + torch.hstack([trainer.datamodule.test_dataset.tiles_unique_selection[idx][i] * trainer.datamodule.test_dataset.options.max_xy * (1. + delta_xy), trainer.datamodule.test_dataset.tiles_min_z[idx][i]]).to(out["recs"][i].device)
                for i in range(len(out["recs"]))
            ], 0)
        else:
            reconstruction = out["recs"][0].view(-1, pl_module.hparams.protos.points, 3)

        recs_k = torch.cat(out["recs_k"], 0)
        recs_k = recs_k.view(-1, 1).repeat(1, pl_module.hparams.protos.points)

        print(recs_k.size(), recs_k.max(), recs_k.min(), recs_k.unique().size())

        
        
        rec = Data(
            pos=reconstruction.flatten(0, 1).detach(),
            point_inst_pred=torch.unique((torch.arange(reconstruction.flatten(0, 1).size(0)) / pl_module.hparams.protos.points).int(), return_inverse=True)[1],
            point_inst=recs_k.flatten(0, 1).detach(),
        )
        color="y;inst_pred;inst;xyz"

        if isearthparserdataset:
            torch.save(Data(
                pos=torch.cat([
                    out["recs"][i].view(-1, pl_module.hparams.protos.points, 3).detach() + torch.hstack([trainer.datamodule.test_dataset.tiles_unique_selection[idx][i] * trainer.datamodule.test_dataset.options.max_xy, trainer.datamodule.test_dataset.tiles_min_z[idx][i]]).to(out["recs"][i].device) for i in range(len(out["recs"]))
                    ], 0).flatten(0, 1).detach().cpu(),
                point_inst=rec.point_inst,
                intensity=torch.index_select(pl_module._protosfeat.squeeze(), 0, rec.point_inst).detach().cpu(),
                point_inst_pred=torch.from_numpy(np.random.permutation(rec.point_inst_pred.max().item()+1)[rec.point_inst_pred.squeeze().cpu().numpy()]),
                point_y=apply_learning_map(torch.index_select(torch.from_numpy(trainer.callbacks[0].best_assign).to(rec.point_inst.device), 0, (torch.arange(pl_module.hparams.protos.points).to(recs_k.device).unsqueeze(0) + recs_k * pl_module.hparams.protos.points).flatten(0, 1).detach().squeeze()), trainer.datamodule.myhparams.learning_map_inv) if pl_module.hparams.name != "superquadrics" else None,
            ).detach().cpu(),
            f"{pl_module.hparams.data.name}_{modelname}_reconstruction.pt")
        else:
            torch.save(Data(
                pos=torch.cat([
                    out["recs"][0].view(-1, pl_module.hparams.protos.points, 3).detach()
                    ], 0).flatten(0, 1).detach().cpu(),
                point_inst=rec.point_inst,
                point_inst_pred=torch.from_numpy(np.random.permutation(rec.point_inst_pred.max().item()+1)[rec.point_inst_pred.squeeze().cpu().numpy()]),
                point_y=torch.index_select(torch.from_numpy(trainer.callbacks[0].best_assign).to(rec.point_inst.device), 0, (torch.arange(pl_module.hparams.protos.points).to(recs_k.device).unsqueeze(0) + recs_k * pl_module.hparams.protos.points).flatten(0, 1).detach().squeeze()) if pl_module.hparams.name != "superquadrics" else None,
            ).detach().cpu(),
            f"{pl_module.hparams.data.name}_{modelname}_reconstruction.pt")
        

        if not issuperquadrics:
            if voxelize:
                #print(rec.pos.device)
                choice = torch.unique((rec.pos / voxelize).int(), return_inverse=True, dim=0)[1]
                for attr in ["point_y", "point_y_pred", "point_inst", "point_inst_pred"]:
                    if hasattr(rec, attr):

                        if getattr(rec, attr).max() < 100:
                            setattr(rec, attr, torch_scatter.scatter_sum(F.one_hot(getattr(rec, attr).squeeze().long()), choice, 0).argmax(-1))
                        else:
                            indices = np.unique(choice.cpu().numpy(), return_index=True)[1]
                            setattr(rec, attr, getattr(rec, attr)[indices])
                for attr in ["pos", "features"]:
                    if hasattr(rec, attr):
                        setattr(rec, attr, torch_scatter.scatter_mean(getattr(rec, attr), choice, 0))
                for attr in ["intensity", "rgbi", "rgb"]:
                    if hasattr(rec, attr):
                        setattr(rec, attr, torch_scatter.scatter_max(getattr(rec, attr), choice, 0)[0])

            
            if rec.pos.size(0) > max_points:
                keep = torch.randperm(rec.pos.size(0))[:max_points]
                for attr in ["point_y", "point_y_pred", "point_inst", "point_inst_pred", "pos", "features", "intensity", "rgbi", "rgb"]:
                    if hasattr(rec, attr):
                        setattr(rec, attr, getattr(rec, attr)[keep])

        if hasattr(trainer.callbacks[0], "best_assign") and pl_module.hparams.name != "superquadrics":
            rec.point_y = torch.index_select(torch.from_numpy(trainer.callbacks[0].best_assign).to(rec.point_inst.device), 0, rec.point_inst.squeeze())

        rec = rec.to("cpu")

        fig2.add_trace(go.Scatter3d(
                x=rec.pos[:, 0].to(datadtype), y=rec.pos[:, 1].to(datadtype), z=rec.pos[:, 2].to(datadtype),
                mode='markers',
                marker=dict(size=ps, color=trainer.datamodule.get_color_from_item(rec, color.split(";")[0])),
                name="reconstruction"
            )
        )
        updatemenus2=[
            dict(
                buttons=list([
                    dict(
                        args=[{"marker": dict(size=ps, color=trainer.datamodule.get_color_from_item(rec, c))}, [0]],
                        label=trainer.datamodule.get_label_from_raw_feature(c),
                        method="restyle"
                    ) for c in color.split(";")
                ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.05,
            xanchor="left",
            y=0.88,
            yanchor="top"
            ),
        ]
        fig2.update_layout(updatemenus=updatemenus2)
        fig2.update_layout(
            scene_aspectmode='data',
        )

        out_string += fig2.to_html(full_html=False, include_plotlyjs='cdn') + "\n<br><br>\n"
        del fig2

        if pl_module.hparams.name != "superquadrics":
            protos = pl_module.get_protos().detach().cpu()
            for transformation, do in pl_module.activated_transformations.items():
                if do:
                    if out[transformation].dim() == 3:
                        out[transformation] = out[transformation].unsqueeze(-1)

                    if "kappa_postsoftmax" in out.keys():
                        if "affine" in transformation:
                            t = (out[transformation] * out["kappa_postsoftmax"][..., 1:].unsqueeze(-1).unsqueeze(-1)).flatten(0, 1).sum(0) / out["kappa_postsoftmax"][..., 1:].flatten(0, 1).sum(0).unsqueeze(-1).unsqueeze(-1)
                        else:
                            t = (out[transformation] * out["kappa_postsoftmax"][..., 1:].unsqueeze(-1)).flatten(0, 1).sum(0) / out["kappa_postsoftmax"][..., 1:].flatten(0, 1).sum(0).unsqueeze(-1)
                    else:
                        assert pl_module.hparams.name == "atlas-net-v2"
                        kappa_postsoftmax = torch.eye(pl_module.hparams.K, device=out[transformation].device).unsqueeze(0)

                        if "affine" in transformation:
                            t = (out[transformation] * kappa_postsoftmax.unsqueeze(-1).unsqueeze(-1)).flatten(0, 1).sum(0) / kappa_postsoftmax.flatten(0, 1).sum(0).unsqueeze(-1).unsqueeze(-1)
                        else:
                            t = (out[transformation] * kappa_postsoftmax.unsqueeze(-1)).flatten(0, 1).sum(0) / kappa_postsoftmax.flatten(0, 1).sum(0).unsqueeze(-1)

                    if transformation.startswith("scale"):
                        protos = protos * t.unsqueeze(1).detach().cpu()
                    elif transformation.startswith("rotY_euler"):
                        import learnableearthparser.utils.rotation as rotation
                        mat = rotation._axis_angle_rotation("Y", math.pi * t.squeeze() / 180).detach().cpu()
                        protos = torch.matmul(protos, mat)
                    elif transformation.startswith("rotXYZ_quat"):
                        import learnableearthparser.utils.rotation as rotation
                        mat = rotation.quaternion_to_matrix(t).detach().cpu()
                        protos = torch.matmul(protos, mat)
                    elif transformation.startswith("affine"):
                        mat = (t + torch.eye(3, device=t.device, dtype=t.dtype).view(1, 3, 3)).detach().cpu()
                        protos = torch.matmul(protos, mat)
                    elif transformation.startswith("rotZ_2d"):
                        pass
                    elif transformation.startswith("translation"):
                        pass
                    else:
                        raise NotImplementedError(transformation)
            protos = protos - protos.mean(1).unsqueeze(1)

            out_string += self.get_protos(protos.numpy())
            if isearthparserdataset:
                torch.save((protos, apply_learning_map(torch.from_numpy(trainer.callbacks[0].best_assign), trainer.datamodule.myhparams.learning_map_inv)), f"{pl_module.hparams.data.name}_{modelname}_prototypes.pt")
            else:
                torch.save((protos, torch.from_numpy(trainer.callbacks[0].best_assign)), f"{pl_module.hparams.data.name}_{modelname}_prototypes.pt")
        else:
            pl_module.SUPERQUADRIC_MODE = "train"
            pl_module.hparams.protos.points = training_Kpoints

        return out_string



    def get_body(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", title):
        body = "<hr>" + self.add_text("h2", title) + self.get_metrics(trainer, pl_module) + self.get_inferences(trainer, pl_module)

        if "test" in title:
            body += self.get_test_scene_inference(trainer, pl_module)
        else:
            body += self.get_protos(pl_module.get_protos().detach().cpu().numpy())
        return body

    @torch.no_grad()
    def do_out_html(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", title="", name=""):
        html_name = f"report{('_' + name) if name is not None else ''}.html"

        if osp.exists(html_name):
            with open(html_name, 'r') as output_file:
                html = output_file.read()
                output_file.close()

            html = html.replace("</body>", self.get_body(trainer, pl_module, title) + "</body>")
        else:
            html = "<!DOCTYPE html>\n<html>\n"
            html += self.get_header(trainer, pl_module)
            html += "<body>\n" + self.get_title(trainer, pl_module) + self.get_body(trainer, pl_module, title) + "</body>\n"
            html += "</html>"

        with open(html_name, 'w') as output_file:
            output_file.write(html)

    @torch.no_grad()
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.do_out_html(trainer, pl_module, "on_train_end", "on_train_end")
        return super().on_train_end(trainer, pl_module)

    @torch.no_grad()
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        self.do_out_html(trainer, pl_module, "on_exception", "on_exception")
        return super().on_exception(trainer, pl_module, exception)

    @torch.no_grad()
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "fit" in pl_module.hparams.load_weights:
            modelname = pl_module.hparams.load_weights.split("fit/")[-1].split("/")[0]
        else:
            modelname = pl_module.hparams.load_weights.split("outputs/")[-1].split("/")[0]
        self.do_out_html(trainer, pl_module, f"{pl_module.hparams.data.name}_{modelname}_on_test_end", f"{pl_module.hparams.data.name}_{modelname}_on_test_end")
        return super().on_test_end(trainer, pl_module)
