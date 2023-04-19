import spconv.pytorch as spconv
from torch import nn

from spconv.core import ConvAlgo

class DownConvLayer(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride, norm=None, key=None, *args, **kwargs):
        super(DownConvLayer, self).__init__()
        self.sparseconv3d = MySparseConv3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, indice_key=key, *args, **kwargs)
        self.norm = getattr(nn, norm)(dim_out)
        self.act = nn.LeakyReLU()

    def forward(self, x):

        x = self.sparseconv3d(x)
        x = x.replace_feature(self.act(self.norm(x.features)))
        return x

class ResidualConvLayer(nn.Module):

    def __init__(self, dim_in, dim_out, norm=None, key=None):
        super(ResidualConvLayer, self).__init__()
        self.conv3x3x3 = conv3x3x3(dim_in, dim_out, indice_key=key)
        self.norm = getattr(nn, norm)(dim_out)
        self.act = nn.LeakyReLU()

    def forward(self, x):

        x_conv = self.conv3x3x3(x)
        x_conv = x_conv.replace_feature(self.act(x.features + self.norm(x_conv.features)))

        return x_conv

class ConvLayer(nn.Module):

    def __init__(self, dim_in, dim_out, norm=None, key=None):
        super(ConvLayer, self).__init__()
        self.conv3x3x3 = conv3x3x3(dim_in, dim_out, indice_key=key)
        self.norm = getattr(nn, norm)(dim_out)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv3x3x3(x)
        x = x.replace_feature(self.act(self.norm(x.features)))

        return x

class Conv1Layer(nn.Module):

    def __init__(self, dim_in, dim_out, norm=None, key=None):
        super(Conv1Layer, self).__init__()
        self.conv1x1x1 = conv1x1x1(dim_in, dim_out, indice_key=key)
        self.norm = getattr(nn, norm)(dim_out)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1x1x1(x)
        x = x.replace_feature(self.act(self.norm(x.features)))

        return x

def conv3x3x3(in_planes, out_planes, indice_key=None):
    return MySubMConv3d(
        in_planes, out_planes, kernel_size=3, stride=1,
        padding=1, bias=True, indice_key=indice_key
    )

def conv1x1x1(in_planes, out_planes, indice_key=None):
    return MySubMConv3d(
        in_planes, out_planes, kernel_size=1, stride=1,
        padding=0, bias=True, indice_key=indice_key
    )

class MySubMConv3d(spconv.SubMConv3d):
    def __repr__(self):
        conv = "x".join([str(ks) for ks in self.kernel_size])
        return f"MySubMConv3d({self.in_channels}, {conv}, {self.out_channels})"

class MySparseConv3d(spconv.SparseConv3d):
    def __repr__(self):
        conv = "x".join([str(ks) for ks in self.kernel_size])
        return f"MySparseConv3d({self.in_channels}, {conv}, {self.out_channels})"

class MySparseInverseConv3d(spconv.SparseInverseConv3d):
    def __repr__(self):
        conv = "x".join([str(ks) for ks in self.kernel_size])
        return f"MySparseInverseConv3d({self.in_channels}, {conv}, {self.out_channels})"