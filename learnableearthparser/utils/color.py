import kornia.color as col

def rgb_to_lab(x):
    return col.rgb_to_lab(x.T.unsqueeze(-1)).squeeze().T / 127.

def lab_to_rgb(x):
    return col.lab_to_rgb(127. * x.T.unsqueeze(-1)).squeeze().T