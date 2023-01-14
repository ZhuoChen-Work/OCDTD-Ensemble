import torch as tr
from torch.nn import Module
from torch.nn.modules import Conv2d
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair


class ConvDist(Conv2d):
    def __init__(self, q, sigma, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', gpu=False):
        super(ConvDist, self).__init__(in_channels, out_channels,
                                       kernel_size, stride, padding,
                                       dilation, groups, bias, padding_mode)
        kernel_size = _pair(kernel_size)
        self.ones = Parameter(tr.ones(1, in_channels //
                                      groups, *kernel_size))
        self.q = q
        self.sigma = sigma

    def forward(self, x):
        x = self.dist(x)
        x = self.hidden(x)
        return x

    def dist(self, x):
        return (F.conv2d(x**2, self.ones, None, self.stride,
                         self.padding, self.dilation, self.groups) +
                F.conv2d(x, self.weight, self.bias,
                         self.stride, self.padding, self.dilation,
                         self.groups) + 1e-2).sqrt()

    def hidden(self, d):
        return (1/(self.q*self.sigma**self.q)) * d**(self.q) - self.alpha.log()

    def set_params(self, centroids, alpha):
        self.weight = Parameter(-2*centroids)
        self.bias = Parameter((centroids**2).sum((1, 2, 3)))
        self.alpha = Parameter(alpha.reshape((1, -1, 1, 1)))


class SoftMinMaxPool(Module):
    def forward(self, x):
        x = self.softmin(x)
        x = self.sum(x)
        return x

    def softmin(self, h):
        return -tr.logsumexp(-h, dim=1, keepdim=True)

    def sum(self, ok):
        return ok.sum(dim=(2, 3), keepdim=True)


class ConvOCDTD(Module):
    def __init__(self, ocsvm, in_channels, stride=1):
        super(ConvOCDTD, self).__init__()
        self.q = ocsvm.q
        self.sigma = ocsvm.sigma
        self.alpha = tr.Tensor(ocsvm.alpha)
        kernel_size = int((ocsvm.svs.shape[1] / in_channels)**.5)
        out_channels = ocsvm.svs.shape[0]
        self.svs = tr.Tensor(ocsvm.svs).to(tr.float)
        self.svs = self.svs.reshape((out_channels, in_channels,
                                     kernel_size, kernel_size))

        self.conv = ConvDist(self.q, self.sigma, in_channels,
                             out_channels, kernel_size, stride=stride)
        self.pool = SoftMinMaxPool()

        self.conv.set_params(self.svs, self.alpha)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

    def explain(self, x):
        d = self.conv.dist(x)
        h = self.conv.hidden(d)
        ok = self.pool.softmin(h)
        out = self.pool.sum(ok)

        h.retain_grad()
        d.retain_grad()
        out.backward()

        RDd = tr.min(ok * h.grad, d/self.q * d.grad)
        del d.grad
        del h.grad
        RDd /= d**2


        C1 = F.conv_transpose2d(RDd,
                                self.conv.ones.expand_as(self.conv.weight),
                                bias=None, stride=self.conv.stride,
                                padding=self.conv.padding,
                                groups=self.conv.groups,
                                dilation=self.conv.dilation)

        C2 = F.conv_transpose2d(RDd, self.conv.weight, bias=None,
                                stride=self.conv.stride,
                                padding=self.conv.padding,
                                groups=self.conv.groups,
                                dilation=self.conv.dilation)

        C3 = F.conv_transpose2d(RDd, 0.25*self.conv.weight**2, bias=None,
                                stride=self.conv.stride,
                                padding=self.conv.padding,
                                groups=self.conv.groups,
                                dilation=self.conv.dilation)

        R = C1*x**2 + C2*x + C3
        return R
