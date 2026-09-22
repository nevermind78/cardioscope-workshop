"""Metrics that a cardiologist (and a reviewer) actually cares about.

Accuracy is misleading here: ~90 % of beats are normal, so a model that
answers "N" all the time is 90 % accurate and clinically useless.  We report
per-class sensitivity (Se) and positive predictive value (PPV, "+P" in the
AAMI EC57 standard), the macro-F1 and calibration measures.
"""
from __future__ import annotations

import numpy as np

from . import config as C

K = len(C.CLASSES)


def confusion_matrix(y_true, y_pred, n: int = K) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def _safe_div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def per_class(y_true, y_pred):
    """Per-class table: beats, Se, PPV, F1, specificity (in %)."""
    import pandas as pd
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn, fp = cm.sum(1) - tp, cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    se, ppv = _safe_div(tp, tp + fn), _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * se * ppv, se + ppv)
    return pd.DataFrame({"Beats": cm.sum(1), "Se (%)": 100 * se, "PPV (%)": 100 * ppv,
                         "F1 (%)": 100 * f1, "Spec (%)": 100 * _safe_div(tn, tn + fp)},
                        index=C.CLASSES).round(1)


def macro_f1(y_true, y_pred) -> float:
    return float(per_class(y_true, y_pred)["F1 (%)"].mean() / 100)


def summary(y_true, y_pred) -> dict:
    """One line per model, handy to compare experiments."""
    t = per_class(y_true, y_pred)
    out = {"Accuracy (%)": round(100 * float(np.mean(np.asarray(y_true) == np.asarray(y_pred))), 1),
           "Macro-F1 (%)": round(float(t["F1 (%)"].mean()), 1)}
    for c in C.CLASSES[1:]:
        out[f"Se {c} (%)"] = float(t.loc[c, "Se (%)"])
        out[f"PPV {c} (%)"] = float(t.loc[c, "PPV (%)"])
    return out


# --------------------------------------------------------------------------
# Probabilities: calibration and selective prediction
# --------------------------------------------------------------------------
def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def reliability(probs, y, n_bins: int = 10):
    """Mean confidence, accuracy and count in each confidence bin."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == np.asarray(y)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(bins, minlength=n_bins)
    mean_conf = _safe_div(np.bincount(bins, conf, n_bins), count)
    acc = _safe_div(np.bincount(bins, correct, n_bins), count)
    return mean_conf, acc, count


def ece(probs, y, n_bins: int = 10) -> float:
    """Expected Calibration Error: average |confidence - accuracy| gap."""
    mean_conf, acc, count = reliability(probs, y, n_bins)
    return float(np.sum(count / count.sum() * np.abs(mean_conf - acc)))


def fit_temperature(logits, y, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al., ICML 2017): one scalar T minimising the
    negative log-likelihood on held-out data.  T > 1 means "the network was
    over-confident"."""
    import torch
    import torch.nn.functional as F
    L = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(L / log_t.exp(), Y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().item())


def risk_coverage(probs, y):
    """Sort beats from most to least confident: error rate among the beats the
    model keeps, as a function of the fraction it keeps (coverage)."""
    conf = probs.max(1)
    order = np.argsort(-conf)
    wrong = (probs.argmax(1) != np.asarray(y))[order].astype(float)
    k = np.arange(1, len(wrong) + 1)
    return k / len(wrong), np.cumsum(wrong) / k, conf[order]


def triage(probs, y, threshold: float) -> dict:
    """Beats below the confidence threshold are referred to a cardiologist."""
    y = np.asarray(y)
    keep = probs.max(1) >= threshold
    wrong = probs.argmax(1) != y

    def pct(x):
        return round(100 * float(np.mean(x)), 1) if np.size(x) else float("nan")

    return {"threshold": threshold,
            "auto-analysed (%)": pct(keep),
            "referred to cardiologist (%)": pct(~keep),
            "error rate, auto-analysed (%)": pct(wrong[keep]),
            "error rate, referred (%)": pct(wrong[~keep]),
            "share of all errors caught (%)": pct(~keep[wrong])}
