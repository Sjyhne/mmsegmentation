# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule
from .decode_head import BaseDecodeHead



class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super(SelfAttention, self).__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels,n_channels,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def _conv(self,n_in,n_out):
        return nn.Conv2d(n_in, n_out, 1, 1, bias=False)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        print("size:", size)
        x = x.view(*size[:2],-1)
        print("x.shape:", x.shape)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = torch.nn.functional.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class DownscaleUpscale(nn.Module):


    def __init__(self):
        super(DownscaleUpscale, self).__init__()

        self.patch_sizes = (8, 4, 2, 1)

        self.cls = nn.Conv2d(4, 2, 1, 1)

        self.attn = MultiheadAttention(128*128, 8)


        self.u1 = nn.ConvTranspose2d(2, 2, 2, 2)
        self.u2 = nn.ConvTranspose2d(4, 2, 2, 2)
        self.u3 = nn.ConvTranspose2d(4, 2, 2, 2)

        self.downscales = torch.nn.ModuleList([nn.Conv2d(2, 2, 2, 2) for i in range(len(self.patch_sizes) - 1)])


    def forward(self, x):

        # x = B x 2 x 512 x 512

        xd1 = self.downscales[0](x)
        xd2 = self.downscales[1](xd1)
        xd3 = self.downscales[2](xd2)

        xu1 = self.u1(xd3)
        xu1 = torch.concat([xu1, xd2], dim=1)

        xu1 = self.attn(xu1.view(*xu1.size()[:2], -1)).view(xu1.size())

        xu2 = self.u2(xu1)
        xu2 = torch.concat([xu2, xd1], dim=1)

        xu3 = self.u3(xu2)
        xu3 = torch.concat([xu3, x], dim=1)

        x = self.cls(xu3)

        return x

