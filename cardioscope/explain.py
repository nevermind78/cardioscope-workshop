"""Opening the black box.

* grad_cam: which part of the beat (P wave, QRS, T wave...) drove the decision?
* rhythm_counterfactual: what would the model say if this beat had arrived
  exactly on time?  If the answer changes, the decision was driven by timing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .training import make_inputs


def _inputs(model, x_store, d_store=None, rr=None):
    device = next(model.parameters()).device
    x = torch.as_tensor(np.asarray(x_store, dtype=np.float32)[None], device=device)
    d = (torch.as_tensor(np.asarray(d_store, dtype=np.float32)[None], device=device)
         if model.use_context else None)
    rb = (torch.as_tensor(np.asarray(rr, dtype=np.float32)[None], device=device)
          if model.use_rr else None)
    return make_inputs(x, d), rb


def grad_cam(model, x_store, d_store=None, rr=None, target=None, temperature: float = 1.0):
    """Grad-CAM (Selvaraju et al., 2017) adapted to 1D signals.

    Returns (cam [WIN] scaled to 0..1, class probabilities [K], target class)."""
    model.eval()
    xb, rb = _inputs(model, x_store, d_store, rr)
    fmap = model.features(xb)                                   # (1, channels, 16)
    logits = model.classify(fmap, rb)
    probs = torch.softmax(logits / temperature, dim=1)[0].detach().cpu().numpy()
    target = int(probs.argmax()) if target is None else int(target)
    (grads,) = torch.autograd.grad(logits[0, target], fmap)
    weights = grads.mean(dim=2, keepdim=True)                   # importance of each feature map
    cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))     # (1, 1, 16)
    cam = F.interpolate(cam, size=C.WIN, mode="linear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, probs, target


@torch.no_grad()
def rhythm_counterfactual(model, x_store, d_store=None, rr=None, temperature: float = 1.0):
    """Probabilities for the real beat and for the same beat arriving on time
    (all RR ratios set to 1).  Only meaningful for models with use_rr=True."""
    if not model.use_rr:
        raise ValueError("The model does not use rhythm features.")
    model.eval()
    xb, rb = _inputs(model, x_store, d_store, rr)
    actual = torch.softmax(model(xb, rb) / temperature, dim=1)[0].cpu().numpy()
    on_time = torch.softmax(model(xb, torch.ones_like(rb)) / temperature, dim=1)[0].cpu().numpy()
    return actual, on_time
