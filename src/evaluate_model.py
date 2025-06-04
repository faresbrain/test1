import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset

from transformations import RandomRotate, RandomTranslate, RandomPermute
from utils import compute_pairwise_distance, load_scaler


class TSPDataset(Dataset):
    def __init__(self, path, max_points=50):
        self.data = json.load(open(path))
        self.max_points = max_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        points = np.asarray(entry["points"], dtype=np.float32)
        dist = compute_pairwise_distance(points, self.max_points)
        score = (
            entry.get("score")
            or entry.get("opt_dist")
            or entry.get("opt_dist_true")
        )
        return dist.flatten(), np.array([score], dtype=np.float32)


def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x).cpu().numpy()
            preds.append(out)
            targets.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    return mean_squared_error(targets, preds)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TSPDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    model = torch.nn.Sequential(
        torch.nn.Linear(50 * 50, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    scaler = load_scaler(args.scaler)
    mse = evaluate(model, dataloader, device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "metrics.json", "w") as f:
        json.dump({"mse": mse}, f, indent=2)
    print(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    main(args)
