---
layout: post
title: "Deep learning project - Vision Transformer part 2"
---

I have decided to run with a segmentation model to run a simple and fast semantic segmentation with ONNX. First we have to preprocess the model in order to load it from PyTorch to ONNX C++. I downloaded the weights from Hugginface here [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer). 

The transformer network architecture looks like this:

<img width="995" height="506" alt="segformer_architecture" src="https://github.com/user-attachments/assets/cb48ee9e-d967-48c9-b3c1-aba5fb4e45ba" />

I will later go through the architecture and dissect the important components. 

After preprocessing the directory with the weights look like this:

```shell
models/segformer/

config.json
labels.json
model.safetensors
preprocess.json
preprocessor_config.json
pytorch_model.bin
README.md
segformer_b0_ade512.onnx
tf_model.h5
``` 

