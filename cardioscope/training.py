"""Training and inference utilities (plain PyTorch, no extra framework)."""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .data import normalize_beats
from .metrics import macro_f1
from .models import CardioNet


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(y, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency weights, softened by `power` (1 = full inverse frequency,
    0 = no weighting).  power=0.5 is a good compromise between sensitivity on
    rare classes and false alarms."""
    counts = np.bincount(np.asarray(y), minlength=len(C.CLASSES)).astype(np.float64)
    w = (counts.max() / np.maximum(counts, 1.0)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def make_inputs(x_store: torch.Tensor, d_store: torch.Tensor | None = None,
                train: bool = False, noise: float = 0.02) -> torch.Tensor:
    """Stored windows (B, 136) -> model inputs (B, channels, 128).

    Channel 0: the beat, z-scored.  Channel 1 (optional): its deviation from the
    patient's dominant beat.  Training: random crop (the R peak moves by up to
    +/- 22 ms) + light noise, so the network does not depend on a perfectly
    placed fiducial point.  Inference: deterministic centre crop."""
    b = x_store.shape[0]
    if train:
        off = torch.randint(0, 2 * C.JITTER + 1, (b, 1), device=x_store.device)
    else:
        off = torch.full((b, 1), C.JITTER, device=x_store.device)
    idx = off + torch.arange(C.WIN, device=x_store.device)[None, :]
    chans = [normalize_beats(torch.gather(x_store, 1, idx))]
    if d_store is not None:
        chans.append(torch.gather(d_store, 1, idx))
    x = torch.stack(chans, dim=1)
    if train and noise > 0:
        x = x + noise * torch.randn_like(x)
    return x


def _batch(model, data_t: dict, idx, device, train: bool):
    """Model inputs for the rows `idx` of a dataset converted to tensors."""
    d = data_t["D"][idx].to(device) if model.use_context else None
    x = make_inputs(data_t["X"][idx].to(device), d, train=train)
    rr = data_t["rr"][idx].to(device) if model.use_rr else None
    return x, rr


def _as_tensors(data: dict) -> dict:
    return {k: torch.from_numpy(data[k]) for k in ("X", "D", "rr", "y") if k in data}


@torch.no_grad()
def predict_logits(model, data: dict, batch_size: int = 4096, device=None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    t = _as_tensors(data)
    out = []
    for i in range(0, len(t["X"]), batch_size):
        xb, rb = _batch(model, t, slice(i, i + batch_size), device, train=False)
        out.append(model(xb, rb).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, len(C.CLASSES)), np.float32)


def predict_proba(model, data: dict, temperature: float = 1.0, **kw) -> np.ndarray:
    logits = torch.from_numpy(predict_logits(model, data, **kw))
    return torch.softmax(logits / temperature, dim=1).numpy()


def train_model(model, train: dict, val: dict | None = None, epochs: int = 12, lr: float = 3e-3,
                batch_size: int = 256, weight_power: float = 0.5, seed: int = 0,
                device=None, verbose: bool = True) -> dict:
    """Train with AdamW + one-cycle schedule and a class-weighted cross-entropy."""
    device = device or get_device()
    seed_everything(seed)
    model.to(device)
    t = _as_tensors(train)
    Y = t["y"]
    n, steps = len(Y), math.ceil(len(Y) / batch_size)
    weights = class_weights(train["y"], weight_power).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.25)
    gen = torch.Generator().manual_seed(seed)
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_macro_f1": [], "seconds": []}
    if verbose:
        print(f"Training on {device} | {n:,} beats | {epochs} epochs | class weights "
              + ", ".join(f"{c}={w:.2f}" for c, w in zip(C.CLASSES, weights.tolist())))
    for ep in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(n, generator=gen)
        for i in range(steps):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            xb, rb = _batch(model, t, idx, device, train=True)
            loss = loss_fn(model(xb, rb), Y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        hist["epoch"].append(ep)
        hist["train_loss"].append(total / n)
        msg = f"epoch {ep:2d}/{epochs} | train loss {total / n:.3f}"
        if val is not None:
            logits = predict_logits(model, val, device=device)
            vl = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(val["y"]),
                                 weight=weights.cpu()).item()
            vf = macro_f1(val["y"], logits.argmax(1))
            hist["val_loss"].append(vl)
            hist["val_macro_f1"].append(vf)
            msg += f" | val loss {vl:.3f} | val macro-F1 {vf:.3f}"
        hist["seconds"].append(time.time() - t0)
        if verbose:
            print(msg + f" | {hist['seconds'][-1]:.0f} s", flush=True)
    model.eval()
    return hist


def save_checkpoint(model, path, temperature: float = 1.0, **meta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in meta.items()}
    torch.save({"state_dict": model.state_dict(), "hparams": dict(model.hparams),
                "temperature": float(temperature), "classes": list(C.CLASSES),
                "fs": C.FS, "win": C.WIN, "meta": clean}, path)
    return path


def load_checkpoint(path, device=None):
    """Returns (model in eval mode, checkpoint dict)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = CardioNet(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
