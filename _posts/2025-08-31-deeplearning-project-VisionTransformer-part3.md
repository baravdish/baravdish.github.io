---
layout: post
title: "Deep learning project - Vision Transformer part 3"
---

Again, looking at the SegFormer model from their [paper](https://arxiv.org/pdf/2105.15203).

First, there is a regular overlapping convolution for getting the token embeddings with patch size `K=7`, stride `S=4` and padding `P=3`. For example we might have:

Input: `[B, 3, 128, 128]` for a batch size `B`. 
Output: `[B, 32, 32, 64]` with `C=64` channels.
Converted to flat tokens: `[B, 1024, 64]`.

This means that adjacent patches overlap by 3 pixels (7-4). It might look something like this in PyTorch:

```python

class OverlapPatchEmbed(nn.Module):

    def __init__(self, in_channels=3, embed_dim=64, k=7, s=4, p=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, k, s, p, bias=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.proj(x)                      # here we get [B,64,H/4,W/4]
        B,C,H4,W4 = x.shape
        x = x.permute(0,2,3,1)                # and now [B,H/4,W/4,64]
        x = self.norm(x)
        tokens = x.reshape(B, H4*W4, C)       # flattening [B,N,64]
        return tokens, (H4, W4)
```

we can also see that the model uses a hierarchial transformer similar to traditional scale-space approaches that progress from coarse-to-fine features, similar to multi-level CNN features.
The hierarchical feature maps are downsampled with the scales (H/4, W/4), (H/8, W/8), ... (H/32,W/32). 

Their Efficient Self-attention module from the paper is:
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/885222c7-1d83-4aee-8841-a3e0b826905f" />

In short: reshape and learn weights of linear layer that reduce the dimension such that the `O(N^2)` becomes `O(N^2 / R)` e.g. original `K`: 1024×64, reshape and learn its reduced form `16x64` the attention is then based on 1024x16 instead of 1024x1024. The reduction is smaller and smaller for each stage.

That was all. This was specifically for Stage 1. The later stages takes feature maps as input instead of image pixels and further spatial downsampling with more channels. So last stage will have full attention as `R=1` but smaller resolution on feature maps.
