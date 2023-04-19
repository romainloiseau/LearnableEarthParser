import torch

@torch.jit.script
def fexp(x, p):
    return torch.sign(x)*(torch.abs(x)**p)

@torch.jit.script
def get_superquadrics(protos: torch.Tensor, epsilons: torch.Tensor, scales: torch.Tensor, etas: torch.Tensor, omegas: torch.Tensor, max_xy: float) -> torch.Tensor:
    a1 = scales[..., 0].unsqueeze(-1)  # size BxMx1
    a2 = scales[..., 1].unsqueeze(-1)  # size BxMx1
    a3 = scales[..., 2].unsqueeze(-1)  # size BxMx1
    e1 = epsilons[..., 0].unsqueeze(-1)  # size BxMx1
    e2 = epsilons[..., 1].unsqueeze(-1)  # size BxMx1

    x = a1 * fexp(torch.cos(etas), e1) * fexp(torch.cos(omegas), e2)
    y = a2 * fexp(torch.cos(etas), e1) * fexp(torch.sin(omegas), e2)
    z = a3 * fexp(torch.sin(etas), e1)

    x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), torch.tensor(1e-6, device=x.device, dtype=x.dtype)) #x.new_tensor(1e-6))
    y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), torch.tensor(1e-6, device=x.device, dtype=x.dtype)) #x.new_tensor(1e-6))
    z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), torch.tensor(1e-6, device=x.device, dtype=x.dtype)) #x.new_tensor(1e-6))
    return max_xy * protos * torch.stack([x, y, z], -1).unsqueeze(2) / protos.max()