import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import REPO_ROOT, load_config
from src.data.pipeline import load_processed, make_datasets
from src.eval.metrics import horizon_metrics
from src.models.losses import masked_mae
from src.models.stgcn.graph import get_adjacency, scaled_laplacian
from src.models.stgcn.model import STGCN


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(cfg: dict, data: dict, device: str) -> STGCN:
    s, w = cfg["stgcn"], cfg["windows"]
    lap = torch.from_numpy(scaled_laplacian(get_adjacency(cfg, data)))
    model = STGCN(cfg["dataset"]["num_nodes"], lap, w["input_len"], w["output_len"],
                  tuple(map(tuple, s["blocks"])), s["kt"], s["ks"], s["dropout"])
    return model.to(device)


@torch.no_grad()
def predict(model, loader, mean, std, device):
    """Run the model over a loader; returns (pred, target, mask) in mph as numpy arrays."""
    model.eval()
    preds, targets, masks = [], [], []
    for x, y, y_mask in loader:
        preds.append((model(x.to(device)) * std + mean).cpu().numpy())
        targets.append(y.numpy())
        masks.append(y_mask.numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)


def run_dir(cfg: dict, tag: str):
    return REPO_ROOT / "results" / tag


def train(cfg: dict, tag: str | None = None, epochs: int | None = None) -> dict:
    s = cfg["stgcn"]
    tag = tag or f"stgcn_{cfg['graph']['adjacency']}"
    out = run_dir(cfg, tag)
    ckpt_dir = REPO_ROOT / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = pick_device()
    data = load_processed(cfg)
    datasets = make_datasets(cfg, data)
    model = build_model(cfg, data, device)
    mean = torch.tensor(data["scaler"].mean, device=device)
    std = torch.tensor(data["scaler"].std, device=device)

    train_loader = DataLoader(datasets["train"], batch_size=s["batch_size"], shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=s["batch_size"])
    test_loader = DataLoader(datasets["test"], batch_size=s["batch_size"])

    opt = torch.optim.Adam(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=s["lr_patience"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] device={device} params={n_params:,} train_windows={len(datasets['train'])}", flush=True)

    history, best_val, best_epoch, stale = [], float("inf"), 0, 0
    best_path = ckpt_dir / "best.pt"
    start = time.time()
    for epoch in range(1, (epochs or s["epochs"]) + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y, y_mask in train_loader:
            x, y, y_mask = x.to(device), y.to(device), y_mask.to(device)
            loss = masked_mae(model(x) * std + mean, y, y_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        pv, tv, mv = predict(model, val_loader, mean, std, device)
        val_mae = horizon_metrics(pv, tv, mv)["average"]["mae"]
        sched.step(val_mae)
        row = {"epoch": epoch, "train_mae": float(np.mean(losses)), "val_mae": val_mae,
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)

        improved = val_mae < best_val - 1e-4
        if improved:
            best_val, best_epoch, stale = val_mae, epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
        print(f"  epoch {epoch:2d} | train {row['train_mae']:.3f} | val {val_mae:.3f} | "
              f"lr {row['lr']:.5f} | {row['seconds']:.0f}s{' *' if improved else ''}", flush=True)
        if stale >= s["patience"]:
            print(f"  early stop: no validation improvement for {s['patience']} epochs", flush=True)
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    pv, tv, mv = predict(model, val_loader, mean, std, device)
    pt, tt, mt = predict(model, test_loader, mean, std, device)
    results = {
        "tag": tag,
        "adjacency": cfg["graph"]["adjacency"],
        "parameters": n_params,
        "device": device,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_minutes": (time.time() - start) / 60,
        "val": horizon_metrics(pv, tv, mv),
        "test": horizon_metrics(pt, tt, mt),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.savez_compressed(out / "test_predictions.npz", pred=pt.astype(np.float32))
    for h, m in results["test"].items():
        print(f"  test {h:>8s}: MAE {m['mae']:.3f} | RMSE {m['rmse']:.3f} | MAPE {m['mape']:.2f}%", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train STGCN on METR-LA")
    parser.add_argument("--adjacency", choices=["distance", "correlation", "identity"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    cfg = load_config()
    if args.adjacency:
        cfg["graph"]["adjacency"] = args.adjacency
    train(cfg, tag=args.tag, epochs=args.epochs)
