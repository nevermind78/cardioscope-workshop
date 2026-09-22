"""From beats to a patient-level arrhythmic risk profile: an "AI Holter".

Pipeline for ANY single-lead ECG (no annotation needed):
    resample + filter -> automatic QRS detection (XQRS) -> beat windows,
    rhythm features and patient context -> CardioNet -> per-beat labels ->
    clinical indicators (PVC burden, runs, bigeminy, SVE activity, pauses).

Thresholds are the ones commonly used in Holter reporting.  This is teaching
material: it is NOT a medical device and must not be used for diagnosis.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .data import extract_windows, patient_context, preprocess, refine_peaks, rr_features
from .training import predict_proba

V_IDX, S_IDX = C.CLASSES.index("V"), C.CLASSES.index("S")


# ==========================================================================
# Reading an arbitrary ECG file
# ==========================================================================
def load_ecg_csv(path):
    """Read an ECG from a text/CSV file.

    One sample per line, or several columns (a strictly increasing first column
    is treated as time and the second column is used).  Header lines are
    skipped; a header such as "Sample Rate,512 hertz" is detected.  Values that
    look like microvolts are converted to millivolts.
    Returns (signal_mV, sampling_rate_found_in_file_or_None)."""
    rows, fs_found = [], None
    with open(path, encoding="utf-8-sig", errors="ignore") as fh:
        for line in fh:
            if ";" in line:                          # European CSV: ';' separator, ',' decimals
                parts = [p.strip().replace(",", ".") for p in line.split(";")]
            else:
                parts = [p.strip() for p in line.replace("\t", ",").split(",")]
            parts = [p for p in parts if p]
            if not parts:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                low = line.lower()
                if fs_found is None and ("sample rate" in low or "sampling" in low):
                    m = re.search(r"(\d+(?:\.\d+)?)", line)
                    if m:
                        fs_found = float(m.group(1))
    if not rows:
        raise ValueError("No numeric samples found in the file.")
    width = Counter(len(r) for r in rows).most_common(1)[0][0]
    arr = np.array([r for r in rows if len(r) == width], dtype=np.float64)
    if arr.shape[1] == 1:
        sig = arr[:, 0]
    else:
        sig = arr[:, 1] if np.all(np.diff(arr[:, 0]) > 0) else arr[:, 0]
    if np.nanmedian(np.abs(sig - np.nanmedian(sig))) > 20:   # microvolts -> millivolts
        sig = sig / 1000.0
    return sig.astype(np.float32), fs_found


# ==========================================================================
# Detection and classification of every beat
# ==========================================================================
def detect_r_peaks(x, fs: int = C.FS) -> np.ndarray:
    """Automatic QRS detection (XQRS algorithm of the wfdb package), followed by
    the same fiducial refinement as the one applied to the training labels."""
    from wfdb import processing
    qrs = processing.xqrs_detect(sig=np.asarray(x, dtype=np.float64), fs=fs, verbose=False)
    peaks = np.unique(refine_peaks(x, np.asarray(qrs, dtype=int), fs))
    if len(peaks) > 1:                                     # 200 ms refractory period
        peaks = peaks[np.r_[True, np.diff(peaks) > int(0.2 * fs)]]
    return peaks


@dataclass
class HolterResult:
    signal: np.ndarray        # preprocessed ECG at C.FS (mV)
    peaks: np.ndarray         # every detected R peak (samples at C.FS)
    beat_idx: np.ndarray      # index in `peaks` of every classified beat
    probs: np.ndarray         # (n_beats, K) calibrated probabilities
    X: np.ndarray             # stored beat windows (for explanations)
    D: np.ndarray             # deviation from the dominant beat
    rr: np.ndarray            # rhythm features
    template: np.ndarray      # the patient's dominant beat
    threshold: float          # confidence threshold for human review
    fs: int = C.FS
    report: dict = field(default_factory=dict)

    @property
    def labels(self):
        return self.probs.argmax(1)

    @property
    def confidence(self):
        return self.probs.max(1)

    @property
    def uncertain(self):
        return self.confidence < self.threshold

    @property
    def positions(self):
        return self.peaks[self.beat_idx]

    def beat_table(self):
        import pandas as pd
        df = pd.DataFrame({"time (s)": np.round(self.positions / self.fs, 3),
                           "class": [C.CLASSES[k] for k in self.labels],
                           "confidence": np.round(self.confidence, 3),
                           "needs review": self.uncertain})
        for k, c in enumerate(C.CLASSES):
            df[f"p({c})"] = np.round(self.probs[:, k], 3)
        return df


def analyze_ecg(sig, fs, model, temperature: float = 1.0, threshold: float = 0.8) -> HolterResult:
    """Full analysis of a single-lead ECG (ideally a lead close to MLII)."""
    x = preprocess(sig, fs)
    if len(x) < 10 * C.FS:
        raise ValueError("At least 10 seconds of ECG are needed.")
    peaks = detect_r_peaks(x)
    if len(peaks) < 5:
        raise ValueError("Fewer than 5 heartbeats detected: check the file and the sampling rate.")
    rr, rr_ok = rr_features(peaks)
    win, win_ok = extract_windows(x, peaks)
    dev = np.zeros_like(win)
    dev[win_ok], template = patient_context(win[win_ok])
    ok = rr_ok & win_ok
    data = {"X": win[ok], "D": dev[ok], "rr": rr[ok]}
    probs = predict_proba(model, data, temperature=temperature)
    res = HolterResult(signal=x, peaks=peaks, beat_idx=np.flatnonzero(ok), probs=probs,
                       X=data["X"], D=data["D"], rr=data["rr"], template=template,
                       threshold=threshold)
    res.report = risk_report(res)
    return res


# ==========================================================================
# Patient-level indicators
# ==========================================================================
def _runs(mask):
    """(start, length) of every run of consecutive True values."""
    m = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return [(int(s), int(e - s)) for s, e in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1))]


def _bigeminy_episodes(is_v, min_cycles: int = 3) -> int:
    """Episodes where every other beat is a PVC for at least `min_cycles` cycles."""
    episodes, i, n = 0, 0, len(is_v)
    while i < n - 1:
        j, cycles = i, 0
        while j + 1 < n and is_v[j] and not is_v[j + 1]:
            cycles, j = cycles + 1, j + 2
        if cycles >= min_cycles:
            episodes, i = episodes + 1, j
        else:
            i += 1
    return episodes


def risk_report(res: HolterResult) -> dict:
    labels, n = res.labels, len(res.labels)
    dur_s = len(res.signal) / res.fs
    hours = dur_s / 3600
    rr_all = np.diff(res.peaks) / res.fs
    hr = 60.0 / rr_all
    hr_smooth = np.convolve(hr, np.ones(8) / 8, mode="valid") if len(hr) >= 8 else hr
    counts = {c: int((labels == k).sum()) for k, c in enumerate(C.CLASSES)}
    pct = {c: 100.0 * counts[c] / max(n, 1) for c in C.CLASSES}
    per_h = {c: counts[c] / hours for c in C.CLASSES}
    pos = res.positions

    def run_rate(start, length):
        return float(60.0 / np.mean(np.diff(pos[start:start + length]) / res.fs))

    v_runs, s_runs = _runs(labels == V_IDX), _runs(labels == S_IDX)
    couplets = sum(1 for _, L in v_runs if L == 2)
    vt = [(s, L, run_rate(s, L)) for s, L in v_runs if L >= 3]
    bigeminy = _bigeminy_episodes(labels == V_IDX)
    longest_s = max((L for _, L in s_runs), default=0)
    s_runs3 = sum(1 for _, L in s_runs if L >= 3)
    longest_pause = float(rr_all.max())
    n_pauses = int((rr_all >= 2.0).sum())
    mean_hr = float(60.0 / rr_all.mean())
    uncertain_pct = 100.0 * float(res.uncertain.mean())
    long_enough = dur_s >= 600          # hourly rates are meaningless on a 1-minute strip
    lown = ("4b" if vt else "4a" if couplets else "2" if (long_enough and per_h["V"] >= 30)
            else "1" if counts["V"] else "0")

    flags = []

    def flag(level, title, detail=""):
        flags.append({"level": level, "title": title, "detail": detail})

    if pct["V"] >= 10:
        flag("high", f"High PVC burden: {pct['V']:.1f} % of beats",
             "A burden of 10 % or more is commonly associated with a risk of PVC-induced "
             "cardiomyopathy; an echocardiogram is usually advised.")
    elif long_enough and per_h["V"] >= 30:
        flag("moderate", f"Frequent PVCs: {per_h['V']:.0f} per hour", "30 or more PVCs per hour.")
    if vt:
        longest = max(vt, key=lambda t: t[1])
        flag("high", f"Ventricular runs: {len(vt)} episode(s) of 3 or more consecutive PVCs",
             f"Longest: {longest[1]} beats at {longest[2]:.0f} bpm. Runs faster than 100 bpm "
             "lasting less than 30 s define non-sustained ventricular tachycardia.")
    if couplets:
        flag("moderate", f"{couplets} ventricular couplet(s)", "Two consecutive PVCs.")
    if bigeminy:
        flag("moderate", f"Ventricular bigeminy: {bigeminy} episode(s)",
             "Every other beat is a PVC for at least 3 cycles.")
    if (long_enough and per_h["S"] >= 30) or longest_s >= 20:
        flag("moderate", "Excessive supraventricular ectopic activity",
             f"{per_h['S']:.0f} SVEs per hour, longest run {longest_s} beats. 30 SVEs per hour "
             "or runs of 20 beats are associated with a higher risk of atrial fibrillation "
             "and stroke (Binici et al., Circulation 2010).")
    if s_runs3:
        flag("moderate", f"{s_runs3} supraventricular run(s) of 3 or more beats",
             "Short runs of atrial tachycardia.")
    if longest_pause >= 3.0:
        flag("high", f"Pause of {longest_pause:.1f} s", "Pauses of 3 s or more warrant clinical review.")
    elif longest_pause >= 2.0:
        flag("moderate", f"Pause of {longest_pause:.1f} s", f"{n_pauses} RR interval(s) of 2 s or more.")
    if mean_hr > 100:
        flag("moderate", f"Tachycardia: mean heart rate {mean_hr:.0f} bpm")
    elif mean_hr < 50:
        flag("moderate", f"Bradycardia: mean heart rate {mean_hr:.0f} bpm")
    if uncertain_pct > 5:
        flag("info", f"{uncertain_pct:.1f} % of beats need a human check",
             f"The model's confidence is below {res.threshold:.2f} for these beats.")
    if hours < 1:
        flag("info", f"Short recording ({dur_s / 60:.1f} min)",
             ("Hourly rates are extrapolated" if long_enough else
              "Criteria based on hourly rates are not assessed below 10 minutes")
             + "; clinical Holter monitoring lasts 24 to 48 h.")

    levels = {f["level"] for f in flags}
    level = "high" if "high" in levels else "moderate" if "moderate" in levels else "low"
    return {"level": level, "duration_s": dur_s, "n_beats": n, "counts": counts,
            "percent": pct, "per_hour": per_h, "mean_hr": mean_hr,
            "min_hr": float(hr_smooth.min()), "max_hr": float(hr_smooth.max()),
            "couplets": couplets, "vt_runs": len(vt), "bigeminy": bigeminy,
            "sve_runs": s_runs3, "longest_sve_run": longest_s,
            "longest_pause_s": longest_pause, "pauses_2s": n_pauses,
            "uncertain_pct": uncertain_pct, "lown_grade": lown, "flags": flags}


def format_report(report: dict) -> str:
    """Plain-text version of the report (for notebooks and logs)."""
    r = report
    lines = [f"Arrhythmic risk profile: {r['level'].upper()}",
             f"  {r['n_beats']} beats in {r['duration_s'] / 60:.1f} min | heart rate "
             f"{r['mean_hr']:.0f} bpm (min {r['min_hr']:.0f}, max {r['max_hr']:.0f})",
             "  " + " | ".join(f"{c}: {r['counts'][c]} ({r['percent'][c]:.1f} %)" for c in C.CLASSES),
             f"  simplified Lown grade: {r['lown_grade']} | longest pause {r['longest_pause_s']:.1f} s"
             f" | beats to review: {r['uncertain_pct']:.1f} %"]
    lines += [f"  [{f['level']}] {f['title']}" for f in r["flags"]]
    return "\n".join(lines)


# ==========================================================================
# Agreement with the cardiologists (MIT-BIH records only)
# ==========================================================================
def _nearest(a, b):
    """For each value of a: index of the nearest value in sorted b, and distance."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(b) == 1:
        return np.zeros(len(a), dtype=int), np.abs(a - b[0])
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    left_closer = np.abs(a - b[idx - 1]) <= np.abs(a - b[idx])
    j = np.where(left_closer, idx - 1, idx)
    return j, np.abs(a - b[j])


def compare_with_reference(res: HolterResult, ann_samples, ann_symbols, fs_ann: int,
                           start_sample: int = 0, tol_s: float = 0.15) -> dict | None:
    """QRS detection accuracy and beat-label agreement with the reference
    annotations (150 ms matching window, as in ANSI/AAMI EC57)."""
    sym = np.asarray(ann_symbols)
    keep = np.isin(sym, list(C.BEAT_SYMBOLS))
    ref = (np.asarray(ann_samples)[keep] - start_sample) * C.FS / fs_ann
    sym = sym[keep]
    inside = (ref >= 0) & (ref < len(res.signal))
    ref, sym = ref[inside], sym[inside]
    if len(ref) == 0:
        return None
    tol = tol_s * C.FS
    _, d_det = _nearest(res.peaks, ref)
    _, d_ref = _nearest(ref, res.peaks)
    j, d = _nearest(res.positions, ref)
    ref_lab = np.array([C.CLASSES.index(C.AAMI_MAP[s]) if s in C.AAMI_MAP else -1 for s in sym[j]])
    ok = (d <= tol) & (ref_lab >= 0)
    return {"qrs_se": float(np.mean(d_ref <= tol)), "qrs_ppv": float(np.mean(d_det <= tol)),
            "n_reference": int(len(ref)), "n_detected": int(len(res.peaks)),
            "y_true": ref_lab[ok], "y_pred": res.labels[ok], "matched_beats": int(ok.sum()),
            "ref_positions": ref, "ref_symbols": sym}
