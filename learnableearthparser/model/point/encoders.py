from torch import nn

def LinearEncoder(dim_in=10, encoder=[128, 128], norm=None):

    layers = []
    for i in range(len(encoder)):
        layers.append(nn.Linear(dim_in if i==0 else encoder[i-1], encoder[i], bias=not norm))
        if norm is not None:
            layers.append(getattr(nn, norm)(encoder[i]))
        layers.append(nn.LeakyReLU())

    return nn.Sequential(*layers)