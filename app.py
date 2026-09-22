"""CardioScope: AI-assisted Holter analysis (Gradio app).

    python app.py                  # local, http://127.0.0.1:7860
    python app.py --share          # public link (e.g. for phones in the room)

Educational demo of the BIOVANCE 2026 workshop. Not a medical device.
"""
from __future__ import annotations

import os

# Conda environments often ship both MKL's and PyTorch's OpenMP runtime
# (two libiomp5md.dll), which aborts at import time with OMP Error #15.
# Must be set before numpy/torch are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import inspect
import tempfile
from pathlib import Path

import numpy as np
import gradio as gr

from cardioscope import config as C
from cardioscope import data as D
from cardioscope import viz
from cardioscope.explain import grad_cam, rhythm_counterfactual
from cardioscope.holter import analyze_ecg, compare_with_reference, load_ecg_csv
from cardioscope.metrics import per_class
from cardioscope.training import load_checkpoint

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "checkpoints" / "cardionet_context.pt"
DATA_DIR = ROOT / "data" / "mitdb"
EXAMPLES = ROOT / "examples"

RECORD_NOTES = {
    100: "normal sinus rhythm, a few APCs", 103: "normal sinus rhythm", 105: "PVCs",
    111: "left bundle branch block", 113: "sinus rhythm, aberrated APCs", 117: "sinus rhythm",
    121: "sinus rhythm", 123: "sinus rhythm, rare PVCs", 200: "PVCs, bigeminy, VT runs",
    202: "atrial fibrillation and flutter episodes", 210: "atrial fibrillation, PVCs, VT runs",
    212: "right bundle branch block", 213: "fusion beats, bigeminy, VT runs",
    214: "LBBB, PVCs, VT runs", 219: "atrial fibrillation, PVCs",
    221: "atrial fibrillation, PVCs, VT runs", 222: "AF, atrial flutter, junctional rhythm",
    228: "PVCs, bigeminy", 231: "RBBB, 2nd-degree AV block",
    232: "sinus bradycardia, frequent APCs, long pauses", 233: "frequent PVCs, bigeminy, VT runs",
    234: "supraventricular tachycardia, junctional beats",
}
RECORD_CHOICES = [f"{r} - {RECORD_NOTES[r]}" for r in C.DS2]
LEVELS = {"high": ("#9B1C1C", "#FCEDED", "High"),
          "moderate": ("#8A5300", "#FDF4E3", "Moderate"),
          "low": ("#1D5E3A", "#EAF6EF", "Low")}
FLAG_COLORS = {"high": "#9B1C1C", "moderate": "#B26A00", "info": "#4B5563"}

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
).set(button_primary_background_fill="#1B2A41", button_primary_background_fill_hover="#2E4266",
      button_primary_text_color="#FFFFFF")


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def risk_card(report: dict) -> str:
    fg, bg, word = LEVELS[report["level"]]
    stats = [
        (f"{report['n_beats']:,}", "beats analysed"),
        (f"{report['mean_hr']:.0f} bpm", f"heart rate ({report['min_hr']:.0f} to {report['max_hr']:.0f})"),
        (f"{report['percent']['V']:.1f} %", "PVC burden"),
        (f"{report['per_hour']['S']:.0f} / h", "supraventricular ectopics"),
        (f"{report['longest_pause_s']:.1f} s", "longest RR interval"),
        (f"{report['uncertain_pct']:.1f} %", "beats to review"),
    ]
    cells = "".join(
        f'<div style="padding:6px 14px 6px 0"><div style="font-size:1.35em;font-weight:600;color:#1B2A41">{v}</div>'
        f'<div style="font-size:.82em;color:#4B5563">{k}</div></div>' for v, k in stats)
    flags = "".join(
        f'<li style="list-style:none;margin:6px 0;padding:4px 10px;border-left:4px solid {FLAG_COLORS[f["level"]]}">'
        f'<div style="font-weight:600;color:#1B2A41">{f["title"]}</div>'
        f'<div style="font-size:.85em;color:#4B5563">{f["detail"]}</div></li>' for f in report["flags"])
    return (
        f'<div style="border:1px solid #E5E7EB;border-radius:10px;padding:16px 18px;background:#FFFFFF">'
        f'<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">'
        f'<span style="font-size:1.05em;color:#1B2A41">Arrhythmic risk profile</span>'
        f'<span style="font-size:1.6em;font-weight:700;color:{fg};background:{bg};padding:2px 12px;'
        f'border-radius:6px">{word}</span>'
        f'<span style="font-size:.85em;color:#4B5563">simplified Lown grade {report["lown_grade"]}</span></div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));margin-top:10px">{cells}</div>'
        f'<ul style="padding:0;margin:10px 0 4px 0">{flags or "<li style=list-style:none>No abnormal finding.</li>"}</ul>'
        f'<div style="font-size:.78em;color:#6B7280;margin-top:8px">Educational demo trained on 22 patients of the '
        f'MIT-BIH Arrhythmia Database. Not a medical device: never use it to make a clinical decision.</div></div>')


def _first_event_time(res) -> float:
    """Start the strip 3 s before the first clinically interesting beat."""
    lab = res.labels
    for cond in (lab == C.CLASSES.index("V"), lab == C.CLASSES.index("S"), res.uncertain):
        idx = np.flatnonzero(cond)
        if len(idx):
            return max(float(res.positions[idx[0]] / res.fs) - 3.0, 0.0)
    return 0.0


def _notable_beat(res) -> int:
    for k in (C.CLASSES.index("V"), C.CLASSES.index("S"), C.CLASSES.index("F")):
        idx = np.flatnonzero(res.labels == k)
        if len(idx):
            return int(idx[np.argmax(res.probs[idx, k])])
    return 0


def explain_beat(res, i: int, temperature: float):
    cam, probs, target = grad_cam(MODEL, res.X[i], res.D[i], res.rr[i], temperature=temperature)
    t = res.positions[i] / res.fs
    fig = viz.plot_gradcam([{"beat": D.center_crop(res.X[i]), "cam": cam, "probs": probs,
                             "title": f"Beat at {t:.2f} s: predicted {C.CLASSES[target]}"}])
    pre = (res.positions[i] - res.peaks[res.beat_idx[i] - 1]) / res.fs
    text = (f"**Beat at {t:.2f} s: {C.CLASS_NAMES[C.CLASSES[target]]}** "
            f"(probability {probs[target]:.2f}). Red areas mark the part of the waveform that "
            f"drove the decision. The beat arrived {pre:.2f} s after the previous one, "
            f"{res.rr[i, 0]:.2f} x this patient's median RR.")
    if MODEL.use_rr:
        _, on_time = rhythm_counterfactual(MODEL, res.X[i], res.D[i], res.rr[i], temperature)
        if abs(on_time[target] - probs[target]) >= 0.2:
            text += (f"\n\nIf the same beat had arrived exactly on time, p({C.CLASSES[target]}) would be "
                     f"{on_time[target]:.2f} instead of {probs[target]:.2f}: the timing matters for this decision.")
        else:
            text += "\n\nArriving on time would barely change the prediction: the shape of the beat drives it."
    return fig, text


def agreement_md(ref) -> str:
    if ref is None:
        return "No reference annotations for this ECG (uploaded file)."
    t = per_class(ref["y_true"], ref["y_pred"])
    t = t[t["Beats"] > 0]
    rows = "\n".join(f"| {c} | {int(r['Beats'])} | {r['Se (%)']:.1f} | {r['PPV (%)']:.1f} |" for c, r in t.iterrows())
    return (f"**QRS detection** (automatic, 150 ms tolerance): sensitivity {100 * ref['qrs_se']:.2f} %, "
            f"positive predictive value {100 * ref['qrs_ppv']:.2f} % "
            f"({ref['n_detected']} detected / {ref['n_reference']} annotated beats).\n\n"
            f"**Beat labels vs the cardiologists** ({ref['matched_beats']} matched beats):\n\n"
            f"| class | beats | sensitivity (%) | PPV (%) |\n|---|---|---|---|\n{rows}")


def _render(res, ref, title):
    start = _first_event_time(res)
    duration = len(res.signal) / res.fs
    strip = viz.plot_holter_strip(res, start, 10.0, ref, title)
    i = _notable_beat(res)
    exp_fig, exp_text = explain_beat(res, i, TEMPERATURE)
    table = res.beat_table()
    csv_path = Path(tempfile.mkdtemp()) / "cardioscope_beats.csv"
    table.to_csv(csv_path, index=False)
    return ({"res": res, "ref": ref, "title": title},
            risk_card(res.report), strip,
            gr.Slider(minimum=0, maximum=max(duration - 10.0, 0.0), value=start, step=0.5),
            viz.plot_tachogram(res), exp_fig, exp_text,
            gr.Number(value=round(float(res.positions[i] / res.fs), 2)),
            agreement_md(ref), table, str(csv_path))


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
def run_mitdb(choice, start_min, dur_min, threshold):
    try:
        record = int(str(choice).split()[0])
        D.download_mitdb(DATA_DIR, [record], verbose=False)
        sig, fs, samples, symbols = D.load_record(record, DATA_DIR)
        a = int(start_min * 60 * fs)
        b = min(len(sig), a + int(dur_min * 60 * fs))
        res = analyze_ecg(sig[a:b], fs, MODEL, TEMPERATURE, threshold)
        ref = compare_with_reference(res, samples, symbols, fs, start_sample=a)
    except (ValueError, RuntimeError, OSError) as err:
        raise gr.Error(str(err))
    title = f"Record {record}, unseen patient ({RECORD_NOTES[record]}), from minute {start_min:g}"
    return _render(res, ref, title)


def run_upload(path, fs, threshold):
    if not path:
        raise gr.Error("Choose an ECG file first (CSV or TXT, one sample per line).")
    try:
        sig, fs_file = load_ecg_csv(path)
        fs_used = float(fs_file or fs)
        if fs_used < 100:
            raise ValueError("The sampling rate must be at least 100 Hz.")
        res = analyze_ecg(sig, fs_used, MODEL, TEMPERATURE, threshold)
    except (ValueError, OSError) as err:
        raise gr.Error(str(err))
    return _render(res, None, f"{Path(path).name} ({fs_used:g} Hz, {len(sig) / fs_used:.0f} s)")


def move_strip(state, start):
    if not state:
        return None
    return viz.plot_holter_strip(state["res"], start, 10.0, state["ref"], state["title"])


def explain_at(state, t):
    if not state:
        raise gr.Error("Analyse an ECG first.")
    res = state["res"]
    i = int(np.argmin(np.abs(res.positions / res.fs - float(t or 0))))
    return explain_beat(res, i, TEMPERATURE)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def build_app(checkpoint=DEFAULT_CKPT) -> gr.Blocks:
    global MODEL, TEMPERATURE
    MODEL, ckpt = load_checkpoint(checkpoint)
    TEMPERATURE = float(ckpt.get("temperature", 1.0))
    blocks_kw = {} if "theme" in inspect.signature(gr.Blocks.launch).parameters else {"theme": THEME}

    with gr.Blocks(title="CardioScope", **blocks_kw) as demo:
        gr.Markdown("# CardioScope\nAI-assisted Holter analysis: every heartbeat is detected, "
                    "classified (N, S, V, F) and summarised into an arrhythmic risk profile.")
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                threshold = gr.Slider(0.5, 0.99, value=0.8, step=0.01, label="Confidence threshold",
                                      info="Beats below it are flagged for a cardiologist.")
                with gr.Tabs():
                    with gr.Tab("MIT-BIH patient"):
                        record = gr.Dropdown(RECORD_CHOICES, value=RECORD_CHOICES[-2], label="Unseen patient")
                        start_min = gr.Slider(0, 29, value=0, step=1, label="Start (minute)")
                        dur_min = gr.Slider(1, 30, value=5, step=1, label="Duration (minutes)")
                        run1 = gr.Button("Analyse this recording", variant="primary")
                    with gr.Tab("Your ECG file"):
                        upload = gr.File(label="ECG file (.csv or .txt)", file_types=[".csv", ".txt"],
                                         type="filepath")
                        fs = gr.Number(value=360, label="Sampling rate (Hz)",
                                       info="Ignored if the file declares its sample rate.")
                        run2 = gr.Button("Analyse this file", variant="primary")
                        examples = sorted(EXAMPLES.glob("*.csv"))
                        if examples:
                            gr.Examples([[str(p), 360] for p in examples], inputs=[upload, fs],
                                        label="Example files (MIT-BIH excerpts)")
            with gr.Column(scale=2):
                card = gr.HTML("<div style='color:#6B7280;padding:24px'>Choose a recording and "
                               "press <b>Analyse</b>: the risk profile appears here.</div>")
        strip = gr.Plot(label="ECG strip (25 mm/s, 10 mm/mV)")
        position = gr.Slider(0, 1790, value=0, step=0.5, label="Strip position (s)")
        tacho = gr.Plot(label="RR tachogram")
        with gr.Row():
            with gr.Column(scale=2):
                exp_plot = gr.Plot(label="Why this label? (Grad-CAM)")
            with gr.Column(scale=1):
                exp_text = gr.Markdown()
                exp_time = gr.Number(value=0, label="Explain the beat nearest to (s)")
                exp_btn = gr.Button("Explain this beat")
        with gr.Accordion("AI vs cardiologists (MIT-BIH records only)", open=False):
            agree = gr.Markdown()
        with gr.Accordion("Beat-by-beat table", open=False):
            table = gr.Dataframe(interactive=False)
            csv_file = gr.File(label="Download (CSV)")

        outputs = [state, card, strip, position, tacho, exp_plot, exp_text, exp_time, agree, table, csv_file]
        run1.click(run_mitdb, [record, start_min, dur_min, threshold], outputs)
        run2.click(run_upload, [upload, fs, threshold], outputs)
        position.release(move_strip, [state, position], strip)
        exp_btn.click(explain_at, [state, exp_time], [exp_plot, exp_text])
    return demo


def launch(demo: gr.Blocks, **kwargs):
    """Launch with the theme wherever this Gradio version expects it."""
    if "theme" in inspect.signature(gr.Blocks.launch).parameters:
        kwargs.setdefault("theme", THEME)
    kwargs.setdefault("allowed_paths", [str(ROOT)])
    return demo.launch(**kwargs)


MODEL, TEMPERATURE = None, 1.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CardioScope Gradio app")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--share", action="store_true", help="create a public *.gradio.live link")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    launch(build_app(args.checkpoint), share=args.share, server_name=args.host, server_port=args.port)
