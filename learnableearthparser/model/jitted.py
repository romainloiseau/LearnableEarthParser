import torch

@torch.jit.script
def inverse_permutations(perms):
    #https://discuss.pytorch.org/t/how-to-quickly-inverse-a-permutation-by-using-pytorch/116205/6
    src = torch.arange(perms.size(-1), device=perms.device)[None].expand(perms.size(0), -1)
    perms_inv = torch.empty_like(perms)
    perms_inv = torch.scatter(perms_inv, dim=1, index=perms, src=src)
    return perms_inv

@torch.jit.script
def voxelize(pc, res: float):
    highres_ind = torch.floor(pc / res).to(torch.int32)
    lowres_ind = torch.unique(highres_ind, dim=0)
    return lowres_ind
    
@torch.jit.script
def compute_iou(ind1, ind2):
    union = torch.unique(torch.cat([ind1, ind2], 0), dim=0).size(0)
    intersection = ((ind1.unsqueeze(0) - ind2.unsqueeze(1)).abs().sum(-1) == 0).sum()
    
    return intersection / union