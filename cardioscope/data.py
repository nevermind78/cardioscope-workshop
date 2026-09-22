"""Data pipeline: PhysioNet download -> filtering -> beats + rhythm features.

The MIT-BIH Arrhythmia Database (Moody & Mark, 2001) is distributed by PhysioNet
under the Open Data Commons Attribution License v1.0.
"""
from __future__ import annotations

import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal as sps

from . import config as C

DEFAULT_DATA_DIR = Path("data/mitdb")
DEFAULT_CACHE = Path("data/mitdb_beats_v2.npz")
MIRRORS = (
    "https://physionet.org/files/mitdb/1.0.0/",               # official PhysioNet server
    "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0/",   # PhysioNet open-data bucket (AWS)
)
_DAT_SIZE = 1_950_000   # every MIT-BIH .dat file: 650 000 samples x 2 leads x 12 bits


# ==========================================================================
# 1. Download
# ==========================================================================
def _is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    return path.suffix != ".dat" or path.stat().st_size == _DAT_SIZE


def _fetch(url: str, dest: Path, timeout: int = 60) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "cardioscope-workshop/1.0"})
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
    tmp.replace(dest)


def download_mitdb(data_dir=DEFAULT_DATA_DIR, records=None, mirrors=MIRRORS,
                   workers: int = 8, verbose: bool = True) -> Path:
    """Download the header (.hea), signal (.dat) and reference annotation (.atr)
    files of the requested MIT-BIH records.  Files already present are skipped,
    so the function is cheap to call again (e.g. after a Colab restart)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    records = C.ALL_RECORDS if records is None else records
    todo = [f"{r}.{ext}" for r in records for ext in ("hea", "dat", "atr")
            if not _is_complete(data_dir / f"{r}.{ext}")]
    if not todo:
        if verbose:
            print(f"MIT-BIH already available in {data_dir} ({len(records)} records).")
        return data_dir

    t0 = time.time()
    for base in mirrors:
        if not todo:
            break
        if verbose:
            print(f"Downloading {len(todo)} files from {base} ...")
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch, base + name, data_dir / name): name for name in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    fut.result()
                except Exception:          # network error, 404, timeout...
                    failed.append(futures[fut])
                if verbose:
                    print(f"\r  {i}/{len(todo)} files", end="", flush=True)
        if verbose:
            print()
        todo = [n for n in failed if not _is_complete(data_dir / n)]

    if todo:
        raise RuntimeError(
            f"{len(todo)} files could not be downloaded (e.g. {todo[:3]}).\n"
            "Plan B: download the ZIP from https://physionet.org/content/mitdb/1.0.0/ "
            "and call cardioscope.data.extract_mitdb_zip('<path to zip>')."
        )
    if verbose:
        print(f"Done in {time.time() - t0:.0f} s -> {data_dir}")
    return data_dir


def extract_mitdb_zip(zip_path, data_dir=DEFAULT_DATA_DIR) -> Path:
    """Plan B (no internet in the room): extract the official PhysioNet ZIP."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"{r}.{ext}" for r in C.ALL_RECORDS for ext in ("hea", "dat", "atr")}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in wanted:
                (data_dir / name).write_bytes(zf.read(member))
                wanted.discard(name)
    if wanted:
        raise RuntimeError(f"Missing in ZIP: {sorted(wanted)[:5]} ...")
    print(f"Extracted {len(C.ALL_RECORDS)} records into {data_dir}")
    return data_dir


# ==========================================================================
# 2. Reading and preprocessing
# ==========================================================================
def load_record(record, data_dir=DEFAULT_DATA_DIR, lead: str = "MLII"):
    """Return (signal_mV, fs, annotation_samples, annotation_symbols).

    The lead is selected by name: in record 114 the MLII lead is the second
    channel, so picking "channel 0" blindly would be a (classic) bug."""
    import wfdb  # imported here so that the rest of the package works without wfdb

    path = str(Path(data_dir) / str(record))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    ch = rec.sig_name.index(lead) if lead in rec.sig_name else 0
    sig = np.asarray(rec.p_signal[:, ch], dtype=np.float32)
    return sig, int(rec.fs), np.asarray(ann.sample), np.asarray(ann.symbol)


def preprocess(sig, fs, target_fs: int = C.FS) -> np.ndarray:
    """Resample to `target_fs` (anti-aliased) then zero-phase band-pass filter."""
    sig = np.nan_to_num(np.asarray(sig, dtype=np.float64))
    if abs(fs - target_fs) > 1e-6:
        frac = Fraction(target_fs / float(fs)).limit_denominator(1000)
        sig = sps.resample_poly(sig, frac.numerator, frac.denominator)
    sos = sps.butter(3, C.BANDPASS, btype="bandpass", fs=target_fs, output="sos")
    return sps.sosfiltfilt(sos, sig).astype(np.float32)


def refine_peaks(sig, peaks, fs: int = C.FS, radius_s: float = 0.05) -> np.ndarray:
    """Move each fiducial point to the largest |deflection| within +/- radius.

    Applied to the cardiologists' annotations (training) *and* to the output of
    the automatic detector (inference): both pipelines then centre the beats on
    exactly the same kind of point."""
    peaks = np.asarray(peaks, dtype=int)
    r = int(round(radius_s * fs))
    out = peaks.copy()
    for i, p in enumerate(peaks):
        a, b = max(p - r, 0), min(p + r + 1, len(sig))
        if b > a:
            out[i] = a + int(np.argmax(np.abs(sig[a:b])))
    return out


def rr_features(peaks, fs: int = C.FS):
    """Rhythm context of every beat, normalised by the patient's own rhythm.

    Returns (features [n, 5], valid [n]).  The first and last beats have no
    previous / next interval, so they are marked invalid."""
    p = np.asarray(peaks, dtype=np.float64)
    n = len(p)
    feats = np.full((n, C.N_RR), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if n < 3:
        return feats, valid
    rr = np.diff(p) / fs                         # rr[k] = interval between beat k and k+1
    med = float(np.median(rr))
    pre = np.r_[np.nan, rr]                      # interval before beat i
    post = np.r_[rr, np.nan]                     # interval after beat i
    # local rhythm = mean of (up to) the 10 intervals that precede `pre`
    csum = np.r_[0.0, np.cumsum(rr)]
    local = np.full(n, med)
    for i in range(2, n):
        lo, hi = max(0, i - 11), i - 1           # intervals rr[lo:hi]
        if hi > lo:
            local[i] = (csum[hi] - csum[lo]) / (hi - lo)
    feats[:, 0] = pre / med
    feats[:, 1] = post / med
    feats[:, 2] = local / med
    feats[:, 3] = pre / local
    feats[:, 4] = post / pre
    valid[1:-1] = True
    feats = np.clip(feats, 0.2, 5.0)
    return feats, valid


def extract_windows(sig, peaks, pre: int = C.PRE_R + C.JITTER, post: int = C.POST_R + C.JITTER):
    """Cut a window around every peak.  Returns (windows [n, pre+post], valid [n])."""
    peaks = np.asarray(peaks, dtype=int)
    valid = (peaks - pre >= 0) & (peaks + post <= len(sig))
    idx = peaks[valid, None] + np.arange(-pre, post)[None, :]
    windows = np.zeros((len(peaks), pre + post), dtype=np.float32)
    windows[valid] = sig[idx]
    return windows, valid


def normalize_beats(x):
    """Per-beat z-score (works for numpy arrays and torch tensors)."""
    if isinstance(x, np.ndarray):
        mu, sd = x.mean(-1, keepdims=True), x.std(-1, keepdims=True)
    else:  # torch.Tensor
        mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True, correction=0)
    return (x - mu) / (sd + 1e-6)


def center_crop(x_store):
    """(n, STORE) stored windows -> (n, WIN) model inputs, normalised."""
    return normalize_beats(np.asarray(x_store)[..., C.JITTER:C.JITTER + C.WIN])


def patient_context(windows):
    """Deviation of every beat from the patient's *dominant beat*.

    Holter software (and cardiologists) judge a beat by comparing it with the
    patient's usual beat.  The dominant beat is the sample-wise median of all
    beats of the recording; the deviation is expressed in "patient units"
    (divided by the dominant beat's own amplitude), which removes most of the
    inter-patient variability (electrode placement, bundle branch block...).
    Returns (deviation [n, STORE], dominant beat [STORE])."""
    w = np.asarray(windows, dtype=np.float32)
    w = w - np.median(w, axis=1, keepdims=True)
    template = np.median(w, axis=0)
    scale = max(float(template[C.JITTER:C.JITTER + C.WIN].std()), 1e-3)
    return ((w - template) / scale).astype(np.float32), template


# ==========================================================================
# 3. Dataset construction
# ==========================================================================
def record_beats(record, data_dir=DEFAULT_DATA_DIR) -> dict:
    """All classifiable beats of one record (reference R positions)."""
    sig, fs, samples, symbols = load_record(record, data_dir)
    x = preprocess(sig, fs)
    is_beat = np.isin(symbols, list(C.BEAT_SYMBOLS))
    samples, symbols = samples[is_beat], symbols[is_beat]
    peaks = refine_peaks(x, np.round(samples * C.FS / fs).astype(int))
    rr, rr_ok = rr_features(peaks)
    win, win_ok = extract_windows(x, peaks)
    dev = np.zeros_like(win)
    dev[win_ok], _ = patient_context(win[win_ok])
    labels = np.array([C.CLASSES.index(C.AAMI_MAP[s]) if s in C.AAMI_MAP else -1 for s in symbols])
    keep = rr_ok & win_ok & (labels >= 0)
    return {
        "X": win[keep], "D": dev[keep], "rr": rr[keep], "y": labels[keep].astype(np.int64),
        "rec": np.full(keep.sum(), int(record), dtype=np.int32),
        "pos": peaks[keep].astype(np.int32), "sym": symbols[keep].astype("<U1"),
    }


def build_dataset(data_dir=DEFAULT_DATA_DIR, cache=DEFAULT_CACHE, records=None,
                  force: bool = False, verbose: bool = True) -> dict:
    """Beats of all records (cached as a compressed .npz after the first run)."""
    cache = Path(cache) if cache else None
    if cache and cache.exists() and not force:
        with np.load(cache) as z:
            data = {k: z[k] for k in z.files}
        for k in ("X", "D"):
            data[k] = data[k].astype(np.float32)
        if verbose:
            print(f"Loaded {len(data['y']):,} beats from cache {cache}")
        return data

    records = C.ALL_RECORDS if records is None else records
    download_mitdb(data_dir, records, verbose=verbose)
    parts = []
    for i, r in enumerate(records, 1):
        parts.append(record_beats(r, data_dir))
        if verbose:
            print(f"\rSegmenting record {r} ({i}/{len(records)})", end="", flush=True)
    if verbose:
        print()
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{**data, "X": data["X"].astype(np.float16),
                                          "D": data["D"].astype(np.float16)})
    if verbose:
        print(f"{len(data['y']):,} beats extracted" + (f" and cached in {cache}" if cache else ""))
    return data


def subset(data: dict, mask) -> dict:
    return {k: v[mask] for k, v in data.items()}


def by_records(data: dict, records) -> dict:
    return subset(data, np.isin(data["rec"], records))


def holdout_split(data: dict, frac: float = 0.1, seed: int = 0):
    """Random stratified split *inside* the same patients (intra-patient!)."""
    rng = np.random.default_rng(seed)
    val = np.zeros(len(data["y"]), dtype=bool)
    for c in np.unique(data["y"]):
        idx = np.flatnonzero(data["y"] == c)
        val[rng.choice(idx, size=max(1, int(round(frac * len(idx)))), replace=False)] = True
    return subset(data, ~val), subset(data, val)


def class_counts(data: dict):
    """Per-class beat counts as a pandas DataFrame row."""
    import pandas as pd
    counts = np.bincount(data["y"], minlength=len(C.CLASSES))
    return pd.Series(counts, index=C.CLASSES)
