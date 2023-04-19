from torch import nn

def LinearDecoder(dim_in=128, decoder=[128, 128], dim_out=20, norm = None, end_with_bias=True):

    layers = []
    for i in range(len(decoder)):
        layers.append(nn.Linear(dim_in if i==0 else decoder[i-1], decoder[i], bias=not norm))
        if norm is not None:
            layers.append(getattr(nn, norm)(decoder[i]))
        layers.append(nn.LeakyReLU())
        
    layers.append(nn.Linear(decoder[-1] if len(decoder)>0 else dim_in, dim_out, bias=end_with_bias))

    return nn.Sequential(*layers)