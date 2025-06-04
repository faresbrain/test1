import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformations import RandomRotate, RandomTranslate, RandomPermute
from utils import compute_pairwise_distance, save_json


class TSPDataset(Dataset):
    def __init__(self, path, max_points=50, augment=False):
        self.data = json.load(open(path))
        self.max_points = max_points
        self.augment = augment
        self.transforms = [RandomRotate(), RandomTranslate(), RandomPermute()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        points = np.asarray(entry["points"], dtype=np.float32)
        if self.augment:
            for t in self.transforms:
                points = t(points)
        dist = compute_pairwise_distance(points, self.max_points)
        score = (
            entry.get("score")
            or entry.get("opt_dist")
            or entry.get("opt_dist_true")
        )
        return dist.flatten(), np.array([score], dtype=np.float32)


def build_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(dataloader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(dataloader.dataset)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TSPDataset(args.dataset, augment=True)
    scaler = StandardScaler()
    all_scores = [(
        d.get("score")
        or d.get("opt_dist")
        or d.get("opt_dist_true")
    ) for d in dataset.data]
    scaler.fit(np.array(all_scores).reshape(-1, 1))
    labels = scaler.transform(np.array(all_scores).reshape(-1, 1)).astype(np.float32)

    for idx, label in enumerate(labels):
        dataset.data[idx]["scaled_score"] = label[0]

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = build_model(50 * 50).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train(model, dataloader, criterion, optimizer, device)
        history.append({"epoch": epoch, "loss": loss})
        print(f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(args.output_dir) / "tsp_invariant_model.pt")
    with open(Path(args.output_dir) / "scaler.pkl", "wb") as f:
        torch.save(scaler, f)
    save_json({"history": history}, Path(args.output_dir) / "metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/train"))
    args = parser.parse_args()
    main(args)
