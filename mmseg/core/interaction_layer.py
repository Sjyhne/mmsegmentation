# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer
from mmseg.core.builder import ITERACTION_LAYERS


@ITERACTION_LAYERS.register_module()
class ChannelExchange(BaseModule):
    """
    channel exchange
    Args:
        p (int, optional): 1/p of the features will be exchanged.
            Defaults to 2.
    """
    def __init__(self, p=2):
        super(ChannelExchange, self).__init__()
        self.p = p

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        
        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        
        return out_x1, out_x2


@ITERACTION_LAYERS.register_module()
class SpatialExchange(BaseModule):
    """
    spatial exchange
    Args:
        p (int, optional): 1/p of the features will be exchanged.
            Defaults to 2.
    """
    def __init__(self, p=2):
        super(SpatialExchange, self).__init__()
        self.p = p

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]
        
        return out_x1, out_x2


@ITERACTION_LAYERS.register_module()
class Aggregation_distribution(BaseModule):
    # Aggregation_Distribution Layer (AD)
    def __init__(self, 
                 channels, 
                 num_paths=2, 
                 attn_channels=None, 
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Aggregation_distribution, self).__init__()
        self.num_paths = num_paths # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)
        
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = build_norm_layer(norm_cfg, attn_channels)[1]
        self.act = build_activation_layer(act_cfg)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)
        return x1 * attn1, x2 * attn2


@ITERACTION_LAYERS.register_module()
class TwoIdentity(BaseModule):
    def __init__(self, *args, **kwargs):
        super(TwoIdentity, self).__init__()

    def forward(self, x1, x2):
        return x1, x2
