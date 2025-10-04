---
layout: post
title: "Deep learning project - Vision Transformer part 4"
---

The decoder is simple. Each feature map from all the four stages is processed through a linear layer and we end up with pixel-wise logits with a common channel size `C` for each map. We then upsample the feature maps to a single size `(H/4, W/4)`, meaning `(H/16, W/16)` we upsample that one x4 and for `(H/32, W/32)` we upsample it x8.
We then concatenate all of them so we should expect to end up with a channel size `4C` of the four stages. Again, we use a linear layer to fuse the channels and go from `4C` to a single channel `C`. 
Finally, we predict the semgentation for `N` classes.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormerDecoder(nn.Module):
    def __init__(self, in_channels=[32, 64, 160, 256], embed_dim=256, num_classes=150):
    
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
    
        
        f4 = f4.permute(0, 2, 3, 1)
        f4 = self.linear_c4(f4)
        f4 = f4.permute(0, 3, 1, 2)
        
        f3 = f3.permute(0, 2, 3, 1)
        f3 = self.linear_c3(f3)
        f3 = f3.permute(0, 3, 1, 2)
        
        f2 = f2.permute(0, 2, 3, 1)
        f2 = self.linear_c2(f2)
        f2 = f2.permute(0, 3, 1, 2)
        
        f1 = f1.permute(0, 2, 3, 1)
        f1 = self.linear_c1(f1)
        f1 = f1.permute(0, 3, 1, 2)
        
        # Upsample all to H/4xW/4
        f4 = F.interpolate(f4, size=(H, W), mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=(H, W), mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)
        # f1 already H/4xW/4
        
        # Concatenate
        f = torch.cat([f4, f3, f2, f1], dim=1)  # [B, 4*embed_dim, H, W]
        
        f = self.linear_fuse(f)
        f = self.bn(f)
        f = F.relu(f)
        
        out = self.linear_pred(f) 
        
        return out
```

This is the mental model I use. I like to think in PyTorch as it is cleaner to sketch mental code and simpler to read. We will see how it translates to C++ and ONNX later on.
Something to note is linear layer + permutation vs 1x1 2D convolutions which probably is faster computationally. 
