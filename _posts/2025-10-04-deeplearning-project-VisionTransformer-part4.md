---
layout: post
title: "Deep learning project - Vision Transformer part 4"
---

The decoder is simple. Each feature map from all the four stages is processed through a linear layer and we end up with pixel-wise logits with a common channel size `C`. We then upsample the feature maps to a single size `(H/4, W/4)`, meaning `(H/16, W/16)` we upsample that one x4 and for `(H/32, W/32)` we upsample it x8.
We then concatenate all of them so we should expect to end up with a channel size `4C` of the four stages. Again, we use a linear layer to fuse the channels and go from `4C` to a single channel `C`. 
Finally, we predict the classes for semgentation be it `N` classes.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormerDecoder(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], embed_dim=256, num_classes=10):
    
        super().__init__()
        
        self.linear_c4 = nn.Linear(in_channels[3], embed_dim)
        self.linear_c3 = nn.Linear(in_channels[2], embed_dim)
        self.linear_c2 = nn.Linear(in_channels[1], embed_dim)
        self.linear_c1 = nn.Linear(in_channels[0], embed_dim)
        
        # 4C to C
        self.linear_fuse = nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_dim)
        
        # C to classes
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        
    def forward(self, features):
    
        f1, f2, f3, f4 = features
        B, _, H, W = f1.shape 
    
        
        _f4 = f4.permute(0, 2, 3, 1)
        _f4 = self.linear_c4(_f4)
        _f4 = _f4.permute(0, 3, 1, 2)
        
        _f3 = f3.permute(0, 2, 3, 1)
        _f3 = self.linear_c3(_f3)
        _f3 = _f3.permute(0, 3, 1, 2)
        
        _f2 = f2.permute(0, 2, 3, 1)
        _f2 = self.linear_c2(_f2)
        _f2 = _f2.permute(0, 3, 1, 2)
        
        _f1 = f1.permute(0, 2, 3, 1)
        _f1 = self.linear_c1(_f1)
        _f1 = _f1.permute(0, 3, 1, 2)
        
        # Upsample all to H/4 x W/4
        _f4 = F.interpolate(_f4, size=(H, W), mode='bilinear', align_corners=False)
        _f3 = F.interpolate(_f3, size=(H, W), mode='bilinear', align_corners=False)
        _f2 = F.interpolate(_f2, size=(H, W), mode='bilinear', align_corners=False)
        # _f1 already H/4 x W/4
        
        # Concatenate
        _f = torch.cat([_f4, _f3, _f2, _f1], dim=1)  # [B, 4*embed_dim, H, W]
        
        _f = self.linear_fuse(_f)
        _f = self.bn(_f)
        _f = F.relu(_f)
        
        out = self.linear_pred(_f) 
        
        return out

This is the mental model we will use. I like to think in PyTorch as it is cleaner to sketch mental code and simpler to read. We will see how it translates to C++ and ONNX later on.
Something to note is linear layer + permutation vs 1x1 2D convolutions which probably is faster computationally. 
