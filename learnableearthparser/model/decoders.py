import torch
import learnableearthparser.utils.rotation as rotation
import math

from ..fast_sampler import fast_sample_on_batch
from ..utils import superquadrics as sq

class Decoders:
    DECODERS_N_PARAMS = {
        "affine_L": 9, "affine_LK": 9,
        "translation_L": 3, "translation_LK": 3,
        "scale_aniso_L": 3, "scale_aniso_LK": 3,
        "scale_iso_L": 1, "scale_iso_LK": 1,
        "rotXYZ_6d_L": 6, "rotXYZ_6d_LK": 6,
        "rotXYZ_euler_L": 3, "rotXYZ_euler_LK": 3,
        "rotZ_2d_L": 2, "rotZ_2d_LK": 2,
        "rotZ_euler_L": 1, "rotZ_euler_LK": 1,
        "rotY_euler_L": 1, "rotY_euler_LK": 1,
        "rotX_euler_L": 1, "rotX_euler_LK": 1,
        "superquadrics_L": 5, "rotXYZ_quat_L": 4,
    }

    @torch.profiler.record_function(f"SHIFT")
    def shift_protos_to_voxel_grid(self, protos, voxel_grid):
        voxel_pos = (voxel_grid[:, 1:] + .5) * self.tile_size
        return protos + voxel_pos.view(-1, 1, 1, 1, 3)

    @torch.profiler.record_function(f"T")
    def get_transformed_protos(self, encoded):
        protos = self.get_protos()

        out = {}

        protos = protos.unsqueeze(0).unsqueeze(0) # K * N * 3 --> 1 * 1 * K * N * 3 for broadcast to slab S and layer L
        

        for transformation, do in self.activated_transformations.items():
            if do:
                protos, out[transformation] = getattr(
                    self, f"do_{transformation}"
                )(
                    encoded.flatten(0, 1), protos
                )

        return protos.contiguous(), out

    def transform_translate(self, params, protos):
        return protos + ((self.tile_size.view(1, 1, 1, 3) * params) / 2.).unsqueeze(-2), params

    def do_superquadrics_L(self, encoded, protos):
        assert self.hparams.K == 1, "Superquadrics only works with K=1"

        params = self.decoders["superquadrics_L"](encoded).view(-1, self.hparams.S, 5)

        epsilons = torch.sigmoid(params[..., :2])*1.1 + 0.4
        scales = torch.sigmoid(params[..., 2:]) * 0.5 + 0.03

        if self.SUPERQUADRIC_MODE == "train":
            etas, omegas = fast_sample_on_batch(scales.detach().cpu().numpy(), epsilons.detach().cpu().numpy(), self.hparams.protos.points)
        elif self.SUPERQUADRIC_MODE == "plot":

            from ..utils.icosphere import generate_icosphere
            v, faces = generate_icosphere(2)

            etas, omegas = torch.asin(v[:, 2] / torch.norm(v, dim=1)), torch.atan2(v[:, 1], v[:, 0])
            etas, omegas = etas.view(1, 1, -1).repeat(int(encoded.shape[0] / self.hparams.S), self.hparams.S, 1).numpy(), omegas.view(1, 1, -1).repeat(int(encoded.shape[0] / self.hparams.S), self.hparams.S, 1).numpy()
        else:
            raise ValueError(f"Unknown SUPERQUADRIC_MODE {self.SUPERQUADRIC_MODE} (should be train or plot)")

        # Make sure we don't get nan for gradients
        etas[etas == 0] += 1e-6
        omegas[omegas == 0] += 1e-6

        # Move to tensors
        etas = scales.new_tensor(etas)
        omegas = scales.new_tensor(omegas)

        protos = sq.get_superquadrics(protos, epsilons, scales, etas, omegas, self.hparams.data.max_xy)

        return protos, params.detach()

    @torch.profiler.record_function(f"TRANSLATE_L")
    def do_translation_L(self, encoded, protos):
        translations = self.decoders["translation_L"](encoded).view(-1, self.hparams.S, 1, 3)

        return self.transform_translate(translations, protos)

    @torch.profiler.record_function(f"TRANSLATE_LK")
    def do_translation_LK(self, encoded, protos):
        translations = self.decoders["translation_LK"](encoded).view(-1, self.hparams.S, self.hparams.K, 3)
        translations = self.hparams.decoders.scales.translation_LK * torch.tanh(translations)
        return self.transform_translate(translations, protos)

    def transform_scale(self, params, protos):
        scales = torch.exp(self.hparams.decoders.scales.scale * torch.tanh(params))

        protos = protos * scales.unsqueeze(-2)
        return protos, scales.detach()

    @torch.profiler.record_function(f"ANISO_LCALE_L")
    def do_scale_aniso_L(self, encoded, protos):
        params = self.decoders["scale_aniso_L"](encoded).view(-1, self.hparams.S, 1, 3)
        return self.transform_scale(params, protos)

    @torch.profiler.record_function(f"ANISO_LCALE_LK")
    def do_scale_aniso_LK(self, encoded, protos):
        params = self.decoders["scale_aniso_LK"](encoded).view(-1, self.hparams.S, self.hparams.K, 3)
        return self.transform_scale(params, protos)

    @torch.profiler.record_function(f"ISO_LCALE_L")
    def do_scale_iso_L(self, encoded, protos):
        params = self.decoders["scale_iso_L"](encoded).view(-1, self.hparams.S, 1, 1)
        return self.transform_scale(params, protos)

    @torch.profiler.record_function(f"ISO_LCALE_LK")
    def do_scale_iso_LK(self, encoded, protos):
        params = self.decoders["scale_iso_LK"](encoded).view(-1, self.hparams.S, self.hparams.K, 1)
        return self.transform_scale(params, protos)

    def transform_affine(self, mat, protos):
        return torch.matmul(protos, mat + torch.eye(3, device=mat.device, dtype=mat.dtype).view(1, 1, 1, 3, 3)), mat.detach()

    @torch.profiler.record_function(f"AFFINE_L")
    def do_affine_L(self, encoded, protos):
        mat = self.decoders["affine_L"](encoded).view(-1, self.hparams.S, 1, 3, 3)
        return self.transform_affine(mat, protos)

    @torch.profiler.record_function(f"AFFINE_LK")
    def do_affine_LK(self, encoded, protos):
        mat = self.decoders["affine_LK"](encoded).view(-1, self.hparams.S, self.hparams.K, 3, 3)
        return self.transform_affine(mat, protos)

    def transform_rot2d(self, params, protos):
        mat = rotation.rotation_2d_to_matrix(params)
        return torch.matmul(protos, mat), params.detach()

    @torch.profiler.record_function(f"ROTZ_2D_LK")
    def do_rotZ_2d_LK(self, encoded, protos):
        params2d = self.decoders["rotZ_2d_LK"](encoded).view(-1, self.hparams.S, self.hparams.K, 2)
        return self.transform_rot2d(params2d, protos)

    @torch.profiler.record_function(f"ROTZ_2D_L")
    def do_rotZ_2d_L(self, encoded, protos):
        params2d = self.decoders["rotZ_2d_L"](encoded).view(-1, self.hparams.S, 1, 2)
        return self.transform_rot2d(params2d, protos)

    def transform_rot6d(self, params, protos):
        mat = rotation.rotation_6d_to_matrix(params)
        return torch.matmul(protos, mat), params.detach()

    @torch.profiler.record_function(f"ROTXYZ_6D_LK")
    def do_rotXYZ_6d_LK(self, encoded, protos):
        params6d = self.decoders["rotXYZ_6d_LK"](encoded).view(-1, self.hparams.S, self.hparams.K, 6)
        return self.transform_rot6d(params6d, protos)

    @torch.profiler.record_function(f"ROTXYZ_6D_L")
    def do_rotXYZ_6d_L(self, encoded, protos):
        params6d = self.decoders["rotXYZ_6d_L"](encoded).view(-1, self.hparams.S, 1, 6)
        return self.transform_rot6d(params6d, protos)
        
    def transform_rotquat(self, params, protos):
        mat = rotation.quaternion_to_matrix(params)
        return torch.matmul(protos, mat), params.detach()

    @torch.profiler.record_function(f"ROTXYZ_quat_L")
    def do_rotXYZ_quat_L(self, encoded, protos):
        params6d = self.decoders["rotXYZ_quat_L"](encoded).view(-1, self.hparams.S, 1, 4)
        return self.transform_rotquat(params6d, protos)
        
    def transform_roteuler(self, params, protos):
        mat = rotation.euler_angles_to_matrix(params)
        return torch.matmul(protos, mat), params.detach()

    @torch.profiler.record_function(f"ROTXYZ_EULER_LK")
    def do_rotXYZ_euler_LK(self, encoded, protos):
        euler_angles = self.decoders["rotXYZ_euler_LK"](encoded).view(-1, self.hparams.S, self.hparams.K, 3)
        return self.transform_roteuler(euler_angles, protos)

    @torch.profiler.record_function(f"ROTXYZ_EULER_L")
    def do_rotXYZ_euler_L(self, encoded, protos):
        euler_angles = self.decoders["rotXYZ_euler_L"](encoded).view(-1, self.hparams.S, 1, 3)
        return self.transform_roteuler(euler_angles, protos)

    def transform_rottheta(self, theta, protos, axis):
        mat = rotation._axis_angle_rotation(axis, theta)
        return torch.matmul(protos, mat), 180 * theta.detach() / math.pi

    @torch.profiler.record_function(f"ROTX_LK")
    def do_rotX_euler_LK(self, encoded, protos):
        theta = self.decoders["rotX_euler_LK"](encoded).view(-1, self.hparams.S, self.hparams.K)
        theta = self.hparams.decoders.scales.theta_LK * math.pi * torch.tanh(theta)
        return self.transform_rottheta(theta, protos, "X")

    @torch.profiler.record_function(f"ROTY_L")
    def do_rotY_euler_L(self, encoded, protos):
        theta = self.decoders["rotY_euler_L"](encoded).view(-1, self.hparams.S, 1)
        theta = self.hparams.decoders.scales.theta_L * math.pi * torch.tanh(theta)
        return self.transform_rottheta(theta, protos, "Y")

    @torch.profiler.record_function(f"ROTY_LK")
    def do_rotY_euler_LK(self, encoded, protos):
        theta = self.decoders["rotY_euler_LK"](encoded).view(-1, self.hparams.S, self.hparams.K)
        theta = self.hparams.decoders.scales.theta_LK * math.pi * torch.tanh(theta)
        return self.transform_rottheta(theta, protos, "Y")

    @torch.profiler.record_function(f"ROTZ_LK")
    def do_rotZ_euler_LK(self, encoded, protos):
        theta = self.decoders["rotZ_euler_LK"](encoded).view(-1, self.hparams.S, self.hparams.K)
        theta = self.hparams.decoders.scales.theta_LK * math.pi * torch.tanh(theta)
        return self.transform_rottheta(theta, protos, "Z")

    @torch.profiler.record_function(f"ROTZ_L")
    def do_rotZ_euler_L(self, encoded, protos):
        theta = self.decoders["rotZ_euler_L"](encoded).view(-1, self.hparams.S, 1)
        theta = self.hparams.decoders.scales.theta_L * math.pi * torch.tanh(theta)
        return self.transform_rottheta(theta, protos, "Z")