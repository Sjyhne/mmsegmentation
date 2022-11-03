# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
# Library imports
import torch

# Code imports
from .resnet import ResNet
from mmcv.runner import BaseModule
from ..builder import BACKBONES

@BACKBONES.register_module()
class DualResNet(BaseModule):


    def __init__(self, depth, **kwargs):
        super(DualResNet, self).__init__()

        self.res1 = ResNet(depth, **kwargs)
        self.res2 = ResNet(depth, **kwargs)

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

        collected_outputs = collected_outputs

        return collected_outputs