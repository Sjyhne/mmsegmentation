# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
# Library imports
import torch

# Code imports
from .resnet import ResNet, ResNetV1c, BasicBlock
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule
from ..builder import BACKBONES

@BACKBONES.register_module()
class TripleResNet(BaseModule):


    def __init__(self, depth, **kwargs):
        super(TripleResNet, self).__init__()

        self.res1 = ResNet(depth, **kwargs)
        self.res2 = ResNet(depth, **kwargs)
        self.bs2 = build_conv_layer(cfg=None, in_channels=2048, out_channels=1024, kernel_size=1)
        self.bs3 = build_conv_layer(cfg=None, in_channels=4096, out_channels=2048, kernel_size=1)

    def forward(self, x):

        x = torch.cat([torch.unsqueeze(xi, dim=0) for xi in x], dim=0)

        original_image = x[:, :3]
        generated_image = x[:, 3:]

        original_output = self.res1(original_image)
        generated_output = self.res2(generated_image)

        collected_outputs = []

        for i in range(len(original_output)):
            tmp = torch.cat([original_output[i], generated_output[i]], dim=1)
            collected_outputs.append(tmp)


        # We do not use bs0 and bs1

        bs0 = collected_outputs[0]
        bs1 = collected_outputs[1]
        bs2 = self.bs2(collected_outputs[2])
        bs3 = self.bs3(collected_outputs[3])

        collected_outputs = [bs0, bs1, bs2, bs3]

        return collected_outputs