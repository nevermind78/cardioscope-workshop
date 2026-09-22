"""CardioNet: a compact 1D residual CNN for heartbeat classification.

  beat (128 samples)            ─┐
  beat - patient's dominant beat ─┴► 1D ResNet ─► global avg pooling ─┐
     (2nd channel only if use_context=True)                           ├─► MLP ─► N / S / V / F
  RR ratios (5 numbers) ─► small MLP (only if use_rr=True) ───────────┘

Three versions are compared in the workshop:
  * morphology only        CardioNet()                          "what does the beat look like?"
  * + rhythm               CardioNet(use_rr=True)               "when did it arrive?"
  * + patient context      CardioNet(use_rr=True, use_context=True)
                                                                "how does it differ from THIS
                                                                 patient's usual beat?"
The last two ingredients are exactly what a cardiologist reading a Holter uses.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class ResBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, k: int = 7, p_drop: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(p_drop)
        self.skip = (nn.Identity() if c_in == c_out and stride == 1 else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                                   nn.BatchNorm1d(c_out)))

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(self.drop(h)))
        return F.relu(h + self.skip(x))


class CardioNet(nn.Module):
    def __init__(self, n_classes: int = len(C.CLASSES), use_rr: bool = False,
                 use_context: bool = False, width: int = 16, p_drop: float = 0.2,
                 seed: int | None = 0):
        super().__init__()
        if seed is not None:          # same initial weights for everyone in the room
            torch.manual_seed(seed)
        self.hparams = {"n_classes": n_classes, "use_rr": use_rr, "use_context": use_context,
                        "width": width, "p_drop": p_drop}
        self.use_rr, self.use_context = use_rr, use_context
        in_ch = 2 if use_context else 1
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, 7, padding=3, bias=False),
                                  nn.BatchNorm1d(width), nn.ReLU())
        self.blocks = nn.Sequential(
            ResBlock1D(width, 2 * width, stride=2),
            ResBlock1D(2 * width, 4 * width, stride=2),
            ResBlock1D(4 * width, 4 * width, stride=2),
        )
        emb = 4 * width
        if use_rr:
            self.rr_net = nn.Sequential(nn.Linear(C.N_RR, 16), nn.ReLU(),
                                        nn.Linear(16, 16), nn.ReLU())
            emb += 16
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(emb, 64), nn.ReLU(),
                                  nn.Dropout(p_drop), nn.Linear(64, n_classes))

    # The forward pass is split in two so that Grad-CAM can grab the feature maps.
    def features(self, x):
        """(B, in_channels, 128) inputs (or (B, 128)) -> (B, channels, 16) feature maps."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.blocks(self.stem(x))

    def classify(self, fmap, rr=None):
        z = fmap.mean(dim=-1)                                   # global average pooling
        if self.use_rr:
            if rr is None:
                raise ValueError("This model needs the RR features (use_rr=True).")
            z = torch.cat([z, self.rr_net(torch.log(rr.clamp(0.2, 5.0)))], dim=1)
        return self.head(z)

    def forward(self, x, rr=None):
        return self.classify(self.features(x), rr)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
