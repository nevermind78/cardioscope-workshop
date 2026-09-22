"""Figures for the notebooks and the app.

All functions return matplotlib `Figure` objects built without pyplot, so they
are thread-safe inside the Gradio app and display automatically in Jupyter /
Colab when they are the last expression of a cell (or with `display(fig)`).
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from . import config as C
from .data import normalize_beats

INK = "#1B2A41"                 # ECG trace
PAPER_BG, PAPER_MINOR, PAPER_MAJOR = "#FFF9F8", "#F7D6D6", "#E9A3A3"
MUTED = "#6B7280"


def _aami(symbol: str) -> str:
    return C.AAMI_MAP.get(symbol, "Q")


def _color(label: str) -> str:
    return C.CLASS_COLORS.get(label, MUTED)


# ==========================================================================
# ECG paper
# ==========================================================================
def ecg_paper(ax, t0, t1, y0, y1):
    """Standard ECG paper: 1 mm = 0.04 s and 0.1 mV (25 mm/s, 10 mm/mV)."""
    ax.set_facecolor(PAPER_BG)
    xs = np.arange(np.ceil(t0 / 0.04) * 0.04, t1, 0.04)
    ys = np.arange(np.ceil(y0 / 0.1) * 0.1, y1, 0.1)
    major_x = np.isclose((xs / 0.2) % 1, 0, atol=1e-6) | np.isclose((xs / 0.2) % 1, 1, atol=1e-6)
    major_y = np.isclose((ys / 0.5) % 1, 0, atol=1e-6) | np.isclose((ys / 0.5) % 1, 1, atol=1e-6)
    ax.vlines(xs[~major_x], y0, y1, colors=PAPER_MINOR, linewidth=0.4, zorder=0)
    ax.hlines(ys[~major_y], t0, t1, colors=PAPER_MINOR, linewidth=0.4, zorder=0)
    ax.vlines(xs[major_x], y0, y1, colors=PAPER_MAJOR, linewidth=0.8, zorder=0)
    ax.hlines(ys[major_y], t0, t1, colors=PAPER_MAJOR, linewidth=0.8, zorder=0)
    ax.set_xlim(t0, t1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(0.4)               # squares stay square
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks(np.arange(np.ceil(t0), t1 + 1e-9, 1.0))
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.set_xlabel("time (s)", color=MUTED, fontsize=8)


def _strip_figure(dur_s, y0, y1, width_in=14.0):
    """Figure whose axes respect the ECG-paper aspect ratio exactly."""
    left, right, bottom, top = 0.15, 0.15, 0.45, 0.35          # margins (inches)
    ax_w = width_in - left - right
    ax_h = ax_w * (y1 - y0) * 0.4 / dur_s
    h = ax_h + bottom + top
    fig = Figure(figsize=(width_in, h), dpi=110)
    ax = fig.add_axes([left / width_in, bottom / h, ax_w / width_in, ax_h / h])
    return fig, ax


def _y_range(seg):
    lo, hi = np.percentile(seg, [0.5, 99.5]) if len(seg) else (-1, 1)
    return max(min(-1.0, lo - 0.35), -3.0), min(max(1.5, hi + 0.75), 4.5)


def plot_record(sig, fs, samples, symbols, start_s=0.0, dur_s=10.0, title=None, show_rr=False):
    """Raw ECG strip with the cardiologists' annotations (Lab 1)."""
    a, b = int(start_s * fs), int((start_s + dur_s) * fs)
    seg = np.asarray(sig[a:b], dtype=float)
    seg = seg - np.median(seg)
    t = start_s + np.arange(len(seg)) / fs
    y0, y1 = _y_range(seg)
    fig, ax = _strip_figure(dur_s, y0, y1)
    ecg_paper(ax, start_s, start_s + dur_s, y0, y1)
    ax.plot(t, seg, color=INK, linewidth=0.9, zorder=2)
    beats = [(s, sym) for s, sym in zip(samples, symbols) if sym in C.BEAT_SYMBOLS and a <= s < b]
    for i, (s, sym) in enumerate(beats):
        lab = _aami(sym)
        ax.text(s / fs, y1 - 0.25, sym, ha="center", va="center", fontsize=9,
                color=_color(lab), fontweight="normal" if lab == "N" else "bold")
        if show_rr and i > 0:
            prev = beats[i - 1][0]
            ax.text((s + prev) / 2 / fs, y0 + 0.2, f"{(s - prev) / fs:.2f}",
                    ha="center", fontsize=7, color=MUTED)
    ax.set_title(title or "", loc="left", fontsize=10, color=INK)
    return fig


def plot_holter_strip(res, start_s=0.0, dur_s=10.0, reference=None, title=None):
    """AI labels above the trace; cardiologists' labels below (if available)."""
    start_s = float(np.clip(start_s, 0, max(len(res.signal) / res.fs - dur_s, 0)))
    a, b = int(start_s * res.fs), int((start_s + dur_s) * res.fs)
    seg = res.signal[a:b]
    t = start_s + np.arange(len(seg)) / res.fs
    y0, y1 = _y_range(seg)
    y0 -= 0.4 if reference is not None else 0
    fig, ax = _strip_figure(dur_s, y0, y1)
    ecg_paper(ax, start_s, start_s + dur_s, y0, y1)
    ax.plot(t, seg, color=INK, linewidth=0.9, zorder=2)
    pos, labels, unc = res.positions, res.labels, res.uncertain
    for p, k, u in zip(pos, labels, unc):
        if a <= p < b:
            lab = C.CLASSES[k]
            ax.text(p / res.fs, y1 - 0.25, lab + ("?" if u else ""), ha="center", va="center",
                    fontsize=9, color=_color(lab), alpha=0.55 if u else 1.0,
                    fontweight="normal" if lab == "N" else "bold", zorder=3)
    ax.text(start_s + 0.02, y1 - 0.08, "AI", fontsize=7, color=MUTED, va="top")
    if reference is not None:
        for p, sym in zip(reference["ref_positions"], reference["ref_symbols"]):
            if a <= p < b:
                lab = _aami(sym)
                ax.text(p / res.fs, y0 + 0.25, lab, ha="center", va="center", fontsize=9,
                        color=_color(lab), fontweight="normal" if lab == "N" else "bold")
        ax.text(start_s + 0.02, y0 + 0.5, "cardiologists", fontsize=7, color=MUTED)
    ax.set_title(title or "", loc="left", fontsize=10, color=INK)
    return fig


def plot_tachogram(res, title="RR tachogram: each dot is one beat"):
    """RR interval preceding each beat over time, coloured by predicted class."""
    fig = Figure(figsize=(14, 3.2), dpi=110, layout="constrained")
    ax = fig.subplots()
    pre = (res.positions - res.peaks[res.beat_idx - 1]) / res.fs
    tmin = res.positions / res.fs / 60
    for k, c in enumerate(C.CLASSES):
        m = res.labels == k
        if m.any():
            ax.scatter(tmin[m], pre[m], s=6 if c == "N" else 14, color=_color(c),
                       alpha=0.45 if c == "N" else 0.9, label=f"{c} ({m.sum()})", linewidths=0)
    ax.axhline(2.0, color=MUTED, linestyle="--", linewidth=0.8)
    ax.text(tmin.min() if len(tmin) else 0, 2.03, "pause threshold (2 s)", fontsize=7, color=MUTED, va="bottom")
    ax.set_ylim(0, min(max(2.4, float(pre.max()) + 0.2 if len(pre) else 2.4), 6.5))
    ax.set_xlabel("time (min)")
    ax.set_ylabel("RR (s)")
    ax.legend(loc="upper right", fontsize=8, ncol=4, frameon=False)
    ax.set_title(title, loc="left", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


# ==========================================================================
# Data exploration (Lab 1)
# ==========================================================================
def plot_class_balance(counts: dict):
    """counts: {"DS1 (train)": Series, "DS2 (test)": Series} -> log-scale bars."""
    fig = Figure(figsize=(7, 3.2), dpi=110, layout="constrained")
    ax = fig.subplots()
    width = 0.8 / len(counts)
    x = np.arange(len(C.CLASSES))
    for i, (name, s) in enumerate(counts.items()):
        bars = ax.bar(x + i * width, s.values, width, label=name, alpha=0.85)
        ax.bar_label(bars, fmt="{:,.0f}", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x + width * (len(counts) - 1) / 2, [f"{c}\n{C.CLASS_NAMES[c]}" for c in C.CLASSES], fontsize=7)
    ax.set_ylabel("beats (log scale)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Heartbeat classes are severely imbalanced", loc="left", fontsize=10)
    return fig


def plot_class_templates(data: dict, n_max: int = 3000, seed: int = 0):
    """Average normalised beat (+/- 1 std) of each class."""
    rng = np.random.default_rng(seed)
    fig = Figure(figsize=(12, 2.8), dpi=110, layout="constrained")
    axes = fig.subplots(1, len(C.CLASSES), sharey=True)
    t = (np.arange(C.WIN) - C.PRE_R) / C.FS * 1000
    for k, (ax, c) in enumerate(zip(axes, C.CLASSES)):
        idx = np.flatnonzero(data["y"] == k)
        idx = rng.choice(idx, size=min(n_max, len(idx)), replace=False)
        x = normalize_beats(data["X"][idx, C.JITTER:C.JITTER + C.WIN])
        mu, sd = x.mean(0), x.std(0)
        ax.fill_between(t, mu - sd, mu + sd, color=_color(c), alpha=0.2, linewidth=0)
        ax.plot(t, mu, color=_color(c), linewidth=1.8)
        ax.axvline(0, color=MUTED, linewidth=0.6, linestyle=":")
        ax.set_title(f"{c}: {C.CLASS_NAMES[c]}", fontsize=8, loc="left")
        ax.set_xlabel("ms from R peak", fontsize=8)
    return fig


def plot_dominant_beat(windows, deviation, template, labels, record=None, seed=0):
    """Lab 4: a patient's beats, their dominant beat, and the deviation channel."""
    rng = np.random.default_rng(seed)
    t = (np.arange(C.STORE) - C.PRE_R - C.JITTER) / C.FS * 1000
    fig = Figure(figsize=(12, 3.2), dpi=110, layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)
    w = windows - np.median(windows, axis=1, keepdims=True)
    normal = np.flatnonzero(labels == 0)
    for i in rng.choice(normal, size=min(60, len(normal)), replace=False):
        ax1.plot(t, w[i], color=_color("N"), alpha=0.08, linewidth=0.8)
    ax1.plot(t, template, color=INK, linewidth=2.2, label="dominant beat (median)")
    for k, c in enumerate(C.CLASSES[1:], start=1):
        idx = np.flatnonzero(labels == k)
        if len(idx):
            ax1.plot(t, w[idx[0]], color=_color(c), linewidth=1.5, label=f"one {c} beat")
            ax2.plot(t, deviation[idx[0]], color=_color(c), linewidth=1.5, label=f"{c} beat")
    ax2.plot(t, deviation[normal[0]], color=_color("N"), linewidth=1.5, label="N beat")
    ax1.set_title(f"Record {record}: beats and dominant beat (mV)" if record else "Beats and dominant beat",
                  loc="left", fontsize=9)
    ax2.set_title("Channel 2 = beat minus dominant beat (patient units)", loc="left", fontsize=9)
    for ax in (ax1, ax2):
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel("ms from R peak", fontsize=8)
    return fig


# ==========================================================================
# Training and evaluation (Labs 2 to 5)
# ==========================================================================
def plot_history(histories: dict):
    fig = Figure(figsize=(11, 3), dpi=110, layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)
    for name, h in histories.items():
        ax1.plot(h["epoch"], h["train_loss"], marker="o", ms=3, label=name)
        if h.get("val_macro_f1"):
            ax2.plot(h["epoch"], h["val_macro_f1"], marker="o", ms=3, label=name)
    ax1.set_title("Training loss", loc="left", fontsize=9)
    ax2.set_title("Macro-F1 on held-out beats of the SAME patients", loc="left", fontsize=9)
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8, frameon=False)
    return fig


def plot_confusions(results: dict):
    """results: {title: (y_true, y_pred)} -> row-normalised confusion matrices."""
    from .metrics import confusion_matrix
    fig = Figure(figsize=(4.2 * len(results), 3.8), dpi=110, layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(results)))
    for ax, (title, (yt, yp)) in zip(axes, results.items()):
        cm = confusion_matrix(yt, yp)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(C.CLASSES)):
            for j in range(len(C.CLASSES)):
                ax.text(j, i, f"{100 * norm[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(C.CLASSES)), C.CLASSES)
        ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("cardiologists")
        ax.set_title(title, fontsize=9)
    return fig


def plot_reliability(probs_by_name: dict, y, n_bins: int = 10):
    from .metrics import ece, reliability
    fig = Figure(figsize=(4.6 * len(probs_by_name), 3.8), dpi=110, layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(probs_by_name)))
    for ax, (name, p) in zip(axes, probs_by_name.items()):
        conf, acc, cnt = reliability(p, y, n_bins)
        m = cnt > 0
        centers = (np.arange(n_bins) + 0.5) / n_bins
        ax.bar(centers[m], acc[m], width=1 / n_bins, edgecolor="white", color="#4A6FA5", alpha=0.8, label="accuracy")
        ax.plot([0, 1], [0, 1], "--", color=MUTED, label="perfect calibration")
        ax.plot(conf[m], acc[m], "o", color="#C0392B", ms=4, label="mean confidence")
        ax.set_title(f"{name}\nECE = {100 * ece(p, y, n_bins):.1f} %", fontsize=9)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    return fig


def plot_risk_coverage(probs, y, thresholds=(0.6, 0.8, 0.9, 0.95)):
    from .metrics import risk_coverage
    cov, risk, conf = risk_coverage(probs, y)
    fig = Figure(figsize=(6.5, 3.6), dpi=110, layout="constrained")
    ax = fig.subplots()
    ax.plot(100 * cov, 100 * risk, color=INK)
    for th in thresholds:
        k = int(np.searchsorted(-conf, -th, side="right"))
        if 0 < k <= len(cov):
            ax.plot(100 * cov[k - 1], 100 * risk[k - 1], "o", color="#C0392B")
            ax.annotate(f"p >= {th}", (100 * cov[k - 1], 100 * risk[k - 1]), fontsize=7,
                        xytext=(-10, 8), textcoords="offset points")
    ax.set_xlabel("beats analysed automatically (%)")
    ax.set_ylabel("error rate on those beats (%)")
    ax.set_title("Refer the least confident beats to a cardiologist", loc="left", fontsize=9)
    return fig


def plot_gradcam(items):
    """items: list of dicts {beat (WIN,), cam (WIN,), probs, target, title}."""
    fig = Figure(figsize=(4 * len(items), 3.0), dpi=110, layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(items)))
    t = (np.arange(C.WIN) - C.PRE_R) / C.FS * 1000
    for ax, it in zip(axes, items):
        beat = it["beat"]
        lo, hi = float(beat.min()) - 0.3, float(beat.max()) + 0.3
        ax.imshow(it["cam"][None, :], extent=[t[0], t[-1], lo, hi], aspect="auto",
                  cmap="Reds", alpha=0.55, vmin=0, vmax=1)
        ax.plot(t, beat, color=INK, linewidth=1.4)
        probs = ", ".join(f"{c} {p:.2f}" for c, p in zip(C.CLASSES, it["probs"]))
        ax.set_title(f"{it.get('title', '')}\n{probs}", fontsize=8, loc="left")
        ax.set_xlabel("ms from R peak", fontsize=8)
        ax.set_yticks([])
    return fig
