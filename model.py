"""The small CNN from notebook 03, plus helpers to run it on a full parking-lot photo.

Kept in one file so notebooks and the demo app all use exactly the same model and preprocessing.
"""
import numpy as np
import torch
import torch.nn as nn

PATCH_SIZE = 64
NAMES = ["empty", "occupied"]


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


def build_model():
    """Same layers, in the same order, as notebook 03 (so the saved weights fit)."""
    return nn.Sequential(
        block(3, 32),
        block(32, 64),
        block(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )


def load_model(path="small_cnn_best.pt", device="cpu"):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def crop_spaces(image, boxes):
    """Cut each [x, y, w, h] box out of a PIL image and resize to 64x64, same as notebook 02.
    Returns a uint8 tensor shaped (N, 3, 64, 64)."""
    patches = []
    for x, y, w, h in boxes:
        patch = image.crop((x, y, x + w, y + h)).resize((PATCH_SIZE, PATCH_SIZE))
        patches.append(np.array(patch))
    return torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).contiguous()


def predict(model, patches, device="cpu"):
    """patches: uint8 (N, 3, 64, 64). Returns (labels, confidence): 0 = empty, 1 = occupied."""
    x = patches.to(device).float() / 255.0
    x = (x - 0.5) / 0.5  # same scaling as training
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu()
    confidence, labels = probs.max(dim=1)
    return labels, confidence
