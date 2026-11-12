#!/usr/bin/env python3
"""
Test DINOv2 / AnyLoc backbone extraction on RTX 4060 GPU
"""
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

print("Loading DINOv2 model (dinov2_vits14)...")
t0 = time.time()
# DINOv2 ViT-S/14 is compact (~85MB), very fast, and gives 384-d feature representation
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = model.to(device).eval()
print(f"Model loaded in {time.time()-t0:.2f}s")

# Test inference on a dummy image or real frame
dummy = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    feat = model(dummy)
print(f"Output feature shape: {feat.shape} (Dimension: {feat.shape[-1]})")

# Benchmark inference speed
t0 = time.time()
n_iters = 50
with torch.no_grad():
    for _ in range(n_iters):
        _ = model(dummy)
torch.cuda.synchronize()
dt = (time.time() - t0) / n_iters * 1000
print(f"Inference latency: {dt:.2f} ms / frame on RTX 4060!")

