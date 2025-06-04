import json
from pathlib import Path

import numpy as np
import torch


def compute_pairwise_distance(points, max_points=50):
    n = len(points)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    if n < max_points:
        padded = np.zeros((max_points, max_points), dtype=np.float32)
        padded[:n, :n] = dist
        return padded
    else:
        return dist[:max_points, :max_points]


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_scaler(path):
    with open(path, "rb") as f:
        return torch.load(f)
