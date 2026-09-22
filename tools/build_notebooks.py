"""Builds the two workshop notebooks (run from the repository root)."""
import nbformat as nbf

REPO = "https://github.com/nevermind78/cardioscope-workshop"

SETUP = r'''# Setup: run this cell first (about 1 minute on Colab)
%matplotlib inline
import os, sys, subprocess
REPO_URL = "https://github.com/nevermind78/cardioscope-workshop.git"   # set once by the instructor
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.isdir("/content/cardioscope-workshop"):
        if "YOUR-ACCOUNT" not in REPO_URL:
            subprocess.run(["git", "clone", "-q", REPO_URL, "/content/cardioscope-workshop"], check=True)
        else:                                   # no GitHub repository: upload the workshop ZIP
            from google.colab import files
            print("Upload cardioscope-workshop.zip (given by the instructor)")
            zip_name = next(iter(files.upload()))
            subprocess.run(["unzip", "-q", "-o", zip_name, "-d", "/content"], check=True)
    os.chdir("/content/cardioscope-workshop")
    if os.getcwd() not in sys.path:                 # os.chdir alone doesn't update sys.path on Colab
        sys.path.insert(0, os.getcwd())
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)

import numpy as np, pandas as pd, torch
from cardioscope import config as C, data as D, viz
from cardioscope.models import CardioNet, count_parameters
from cardioscope.training import get_device, train_model, predict_logits, save_checkpoint, load_checkpoint
from cardioscope.metrics import per_class, summary
pd.set_option("display.width", 200)
DEVICE = get_device()
EPOCHS = int(os.environ.get("CARDIOSCOPE_EPOCHS", 12))
gpu = f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else " (no GPU: training takes ~3 min per model)"
print(f"PyTorch {torch.__version__} | device: {DEVICE}{gpu}")'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def notebook(cells, path):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"}}
    nbf.write(nb, path)
    print("wrote", path, len(cells), "cells")


# ==========================================================================
# PART 1
# ==========================================================================
p1 = [
md(f'''
# CardioScope, Part 1: from raw ECG to beat-by-beat diagnosis

**BIOVANCE 2026, École Polytechnique de Sousse**. Workshop *Deep Learning for Cardiovascular Risk Detection*.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-ACCOUNT/cardioscope-workshop/blob/main/Part1_ECG_to_Diagnosis.ipynb)

Today we build **CardioScope**, an assistant that reads a Holter ECG the way a cardiologist does:
it finds every heartbeat, labels it, and turns the labels into a patient-level arrhythmic risk profile.

This morning we build the beat classifier and, above all, we learn to **evaluate it honestly**.
This afternoon (Part 2) we turn it into a tool a clinician could trust.

| Lab | Question |
|---|---|
| 1 | What does an annotated ECG look like? |
| 2 | Can a 1D CNN learn to recognise abnormal beats? |
| 3 | Does it work on patients it has never seen? |
| 4 | What does a cardiologist use that our CNN ignores? |

**Colab tip:** *Runtime > Change runtime type > T4 GPU*. Without a GPU everything still runs, just slower.
'''),
code(SETUP),
md('''
## Lab 1. Meet the data: the MIT-BIH Arrhythmia Database

* 48 half-hour ambulatory ECG recordings from 47 subjects (Beth Israel Hospital, Boston, 1975-1979), 2 leads, 360 Hz.
* Every beat was annotated independently by two or more cardiologists (about 110,000 labels).
* Freely available on PhysioNet (Open Data Commons Attribution License).

The download takes 1 to 2 minutes (about 90 MB; Colab downloads it, not your laptop).
'''),
code(r'''
D.download_mitdb()
sig, fs, samples, symbols = D.load_record(208)
print(f"Record 208: {len(sig) / fs / 60:.1f} min at {fs} Hz, {len(samples)} annotations")
pd.Series(symbols).value_counts().head(8).to_frame("count").T
'''),
code(r'''
viz.plot_record(sig, fs, samples, symbols, start_s=60, dur_s=10, show_rr=True,
                title="Record 208, lead MLII. Letters: cardiologists' labels. Numbers: RR intervals (s)")
'''),
md('''
The labels follow the **ANSI/AAMI EC57** standard, which groups the beat types into 5 classes:

| AAMI class | Meaning | MIT-BIH symbols |
|---|---|---|
| **N** | normal or bundle branch block beat | N, L, R, e, j |
| **S** | supraventricular ectopic beat (premature atrial / junctional) | A, a, J, S |
| **V** | ventricular ectopic beat (PVC) | V, E |
| **F** | fusion of a ventricular and a normal beat | F |
| Q | paced or unclassifiable (excluded, as in most studies) | /, f, Q |

**Your turn (1 minute):** in the strip above, which beats look different? Which ones arrive early?
'''),
md('''
## From recordings to a beat dataset

Pipeline (see `cardioscope/data.py`): resample to 180 Hz, band-pass filter 0.5 to 40 Hz, cut a window of
0.25 s before and 0.46 s after each R peak, and compute the rhythm context of each beat.

**The golden rule (de Chazal et al., IEEE TBME 2004):** 22 patients (DS1) for training, 22 *other* patients (DS2) for testing.
Beats of one patient never appear on both sides.
'''),
code(r'''
data = D.build_dataset()
ds1, ds2 = D.by_records(data, C.DS1), D.by_records(data, C.DS2)
counts = {"DS1 (train)": D.class_counts(ds1), "DS2 (test)": D.class_counts(ds2)}
pd.DataFrame(counts).T
'''),
code(r'''viz.plot_class_balance(counts)'''),
code(r'''viz.plot_class_templates(ds1)'''),
md('''
**Quick quiz.** A "model" that always answers *N*: what accuracy does it get on DS2?
'''),
code(r'''
always_n = np.zeros_like(ds2["y"])
print(f"Accuracy of 'always N': {100 * np.mean(always_n == ds2['y']):.1f} %")
per_class(ds2["y"], always_n)
'''),
md('''
Almost 90 % accuracy, and it misses every single abnormal beat. **Accuracy lies on imbalanced data.**
From now on we report, for each class, the **sensitivity** (Se: share of the real beats that are found) and the
**positive predictive value** (PPV: share of the alarms that are real), as required by AAMI EC57.

## Lab 2. A first 1D CNN: morphology only

`CardioNet` is a compact 1D ResNet (3 residual blocks, about 125,000 parameters) that looks at the shape of one beat.
To monitor training we keep 10 % of the DS1 beats aside, drawn at random. Keep in mind that these validation beats come
from **the same 22 patients** as the training beats.
'''),
code(r'''
train_set, val_set = D.holdout_split(ds1, frac=0.1, seed=0)
model_m = CardioNet()
print(model_m.stem, "...", f"\n{count_parameters(model_m):,} trainable parameters")
'''),
code(r'''hist_m = train_model(model_m, train_set, val_set, epochs=EPOCHS, device=DEVICE)'''),
code(r'''viz.plot_history({"morphology": hist_m})'''),
md('''
## Lab 3. The leakage trap

Validation looks excellent. Now the real question: **what happens with 22 patients the network has never seen?**
'''),
code(r'''
pred_val = predict_logits(model_m, val_set).argmax(1)
pred_test = predict_logits(model_m, ds2).argmax(1)
pd.DataFrame({"same patients (validation)": summary(val_set["y"], pred_val),
              "NEW patients (DS2)": summary(ds2["y"], pred_test)}).T
'''),
code(r'''
viz.plot_confusions({"Validation: patients seen in training": (val_set["y"], pred_val),
                     "Test: 22 new patients": (ds2["y"], pred_test)})
'''),
code(r'''per_class(ds2["y"], pred_test)'''),
md('''
**What happened?** The network learned to recognise *patients* (electrode placement, individual QRS shape)
rather than *arrhythmias*. With a random split of the beats, the same patient sits in training and in test, and the
scores look spectacular. Many published "99 % accuracy" results come from exactly this mistake.

> **Rule:** in medical AI, split by patient, never by sample.

## Lab 4. Think like a cardiologist

A cardiologist reading a Holter uses two things our CNN ignores.

**1. Timing.** A premature atrial beat often has a perfectly normal shape: it is only *early*.
We give the network 5 rhythm features, each normalised by the patient's own rhythm:
'''),
code(r'''
print("\n".join(f"  {i + 1}. {f}" for i, f in enumerate(C.RR_FEATURES)))
sig, fs, samples, symbols = D.load_record(220)
t_a = samples[symbols == "A"][0] / fs
viz.plot_record(sig, fs, samples, symbols, start_s=t_a - 4, dur_s=10, show_rr=True,
                title="Record 220: the 'A' beat looks normal, but it arrives early")
'''),
code(r'''
model_r = CardioNet(use_rr=True)
hist_r = train_model(model_r, train_set, val_set, epochs=EPOCHS, device=DEVICE)
'''),
md('''
**2. The patient's usual beat.** A cardiologist judges a beat against *this patient's* normal beats: a wide QRS is
alarming in one patient and routine in a patient with bundle branch block. We compute each patient's **dominant beat**
(the median of all their beats) and give the network a second channel: *beat minus dominant beat*.
'''),
code(r'''
r = D.by_records(data, [208])
deviation, dominant = D.patient_context(r["X"])
viz.plot_dominant_beat(r["X"], deviation, dominant, r["y"], record=208)
'''),
code(r'''
model_c = CardioNet(use_rr=True, use_context=True)
hist_c = train_model(model_c, train_set, val_set, epochs=EPOCHS, device=DEVICE)
'''),
md('''
### The verdict on the 22 unseen patients
'''),
code(r'''
models = {"1. morphology": model_m, "2. + rhythm": model_r, "3. + patient context": model_c}
preds = {name: predict_logits(m, ds2).argmax(1) for name, m in models.items()}
pd.DataFrame({name: summary(ds2["y"], p) for name, p in preds.items()}).T
'''),
code(r'''
viz.plot_confusions({name: (ds2["y"], preds[name]) for name in ("1. morphology", "3. + patient context")})
'''),
md('''
### Honest limits: look at a few patients
'''),
code(r'''
def per_record(pred, records):
    rows = {}
    for rec in records:
        m = ds2["rec"] == rec
        rows[rec] = {**{f"true {c}": int((ds2["y"][m] == k).sum()) for k, c in enumerate(C.CLASSES)},
                     **{f"predicted {c}": int((pred[m] == k).sum()) for k, c in enumerate(C.CLASSES)}}
    return pd.DataFrame(rows).T

per_record(preds["3. + patient context"], [214, 219, 232, 233])
'''),
md('''
* **233** (frequent PVCs) and **214** (left bundle branch block + PVCs): almost perfect.
* **232**: most beats are premature atrial beats conducted with a right bundle branch block. They are the majority,
  so they *define* the patient's median rhythm and dominant beat, and our two "cardiologist" features fail.
* **219**: atrial fibrillation. The rhythm is irregular by nature, so many normal beats look premature (false S).
  The model has no notion of AF: this is a *rhythm* problem, not a *beat* problem.

The detection of supraventricular beats in unseen patients remains an open research problem.

## Save your models for this afternoon
'''),
code(r'''
os.makedirs("checkpoints", exist_ok=True)
for name, m in (("morph", model_m), ("rhythm", model_r), ("context", model_c)):
    print("saved", save_checkpoint(m, f"checkpoints/my_cardionet_{name}.pt", epochs=EPOCHS))
# Colab resets the machine after a long break. To keep your models, copy them to Google Drive:
# from google.colab import drive; drive.mount("/content/drive")
# !mkdir -p /content/drive/MyDrive/cardioscope && cp checkpoints/my_*.pt /content/drive/MyDrive/cardioscope/
'''),
md('''
## Challenge for the break

Change **one** thing, retrain, and compare on DS2 (never tune on DS2 in real research, use a patient-wise validation!).
Ideas: `weight_power=1.0` (stronger class weights), `width=32`, `epochs=20`, `lr=1e-3`.
'''),
code(r'''
my_model = CardioNet(use_rr=True, use_context=True, width=16)
my_hist = train_model(my_model, train_set, val_set, epochs=EPOCHS, weight_power=0.5, device=DEVICE, verbose=False)
pd.DataFrame({"my model": summary(ds2["y"], predict_logits(my_model, ds2).argmax(1)),
              "3. + patient context": summary(ds2["y"], preds["3. + patient context"])}).T
'''),
]

# ==========================================================================
# PART 2
# ==========================================================================
p2 = [
md('''
# CardioScope, Part 2: from model to clinical tool

**BIOVANCE 2026, École Polytechnique de Sousse**. Workshop *Deep Learning for Cardiovascular Risk Detection*.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-ACCOUNT/cardioscope-workshop/blob/main/Part2_Model_to_Clinic.ipynb)

This morning we built a beat classifier and evaluated it honestly. A good score is not enough for a clinical tool:

| Lab | Question |
|---|---|
| 5 | Can we trust its probabilities? Does it know when it does not know? |
| 6 | Why does it take a decision? |
| 7 | How do beat labels become a patient-level risk profile? |
| Deploy | Can a clinician use it without writing code? |
'''),
code(SETUP + r'''
from cardioscope.metrics import ece, fit_temperature, softmax, triage
from cardioscope.explain import grad_cam, rhythm_counterfactual
from cardioscope.holter import analyze_ecg, compare_with_reference, format_report'''),
md('''
## Catch-up: data and models

Your own models from this morning are used if they are still there; otherwise the pretrained ones shipped with the workshop.
'''),
code(r'''
D.download_mitdb()
data = D.build_dataset()
ds1, ds2 = D.by_records(data, C.DS1), D.by_records(data, C.DS2)
train_set, val_set = D.holdout_split(ds1, frac=0.1, seed=0)
cal, ev = D.by_records(data, C.CAL_RECORDS), D.by_records(data, C.EVAL_RECORDS)

def load(name):
    mine = f"checkpoints/my_cardionet_{name}.pt"
    path = mine if os.path.exists(mine) else f"checkpoints/cardionet_{name}.pt"
    model, _ = load_checkpoint(path, DEVICE)
    print("loaded", path)
    return model

model_m, model_r, model_c = load("morph"), load("rhythm"), load("context")
'''),
md('''
## Lab 5. An AI that knows when it doesn't know

When the model says "V with probability 0.95", is it right 95 % of the time? The gap between confidence and accuracy is
the **Expected Calibration Error (ECE)**. *Temperature scaling* (Guo et al., ICML 2017) fixes it with one number T:
probabilities = softmax(logits / T).

We split the 22 unseen patients in two cohorts: 11 **calibration** patients (to fit T) and 11 **evaluation** patients.
'''),
code(r'''
cache, rows = {}, {}
for name, m in (("morphology", model_m), ("+ rhythm", model_r), ("patient context", model_c)):
    lv, lc, le = (predict_logits(m, d) for d in (val_set, cal, ev))
    t_same, t_new = fit_temperature(lv, val_set["y"]), fit_temperature(lc, cal["y"])
    cache[name] = {"logits_eval": le, "T": t_new,
                   "gain": ece(softmax(le), ev["y"]) - ece(softmax(le, t_new), ev["y"])}
    rows[name] = {"ECE, no calibration (%)": 100 * ece(softmax(le), ev["y"]),
                  "T fitted on same patients": t_same,
                  "ECE with that T (%)": 100 * ece(softmax(le, t_same), ev["y"]),
                  "T fitted on new patients": t_new,
                  "ECE with this T (%)": 100 * ece(softmax(le, t_new), ev["y"])}
pd.DataFrame(rows).T.round(2)
'''),
code(r'''
name = max(cache, key=lambda k: cache[k]["gain"])        # the most striking case
le, T_new = cache[name]["logits_eval"], cache[name]["T"]
viz.plot_reliability({f"model '{name}', T = 1": softmax(le),
                      f"T = {T_new:.2f} fitted on new patients": softmax(le, T_new)}, ev["y"])
'''),
md('''
**Lesson:** on the patients it was trained on, the network looks well calibrated (T close to 1). On new patients it is
**over-confident**, and only a calibration done on *new* patients reveals and corrects it. Calibration, like evaluation,
must be inter-patient.

### Triage: refer the uncertain beats to a cardiologist
'''),
code(r'''
T = cache["patient context"]["T"]
probs_ev = softmax(cache["patient context"]["logits_eval"], T)
pd.DataFrame([triage(probs_ev, ev["y"], th) for th in (0.6, 0.8, 0.9, 0.95)])
'''),
code(r'''viz.plot_risk_coverage(probs_ev, ev["y"])'''),
md('''
Beats referred to the cardiologist are several times more likely to be wrong than the beats the model keeps.
**Your turn:** which threshold would you choose for a screening tool? For a tool used at night without a doctor?

## Lab 6. Opening the black box

**Grad-CAM** colours the part of the beat that drove the decision (red = important).
'''),
code(r'''
probs_ds2 = softmax(predict_logits(model_c, ds2), T)
pred = probs_ds2.argmax(1)
items = []
for c in ("N", "S", "V"):
    k = C.CLASSES.index(c)
    idx = np.flatnonzero((ds2["y"] == k) & (pred == k))
    i = idx[np.argmax(probs_ds2[idx, k])]
    cam, p, _ = grad_cam(model_c, ds2["X"][i], ds2["D"][i], ds2["rr"][i], temperature=T)
    items.append({"beat": D.center_crop(ds2["X"][i]), "cam": cam, "probs": p,
                  "title": f"Record {ds2['rec'][i]}: a {c} beat"})
viz.plot_gradcam(items)
'''),
md('''
**Counterfactual question:** what if the premature beat had arrived exactly on time? If the probability of S collapses,
the decision was driven by **timing**, exactly like a cardiologist's.
'''),
code(r'''
k = C.CLASSES.index("S")
idx = np.flatnonzero((ds2["y"] == k) & (pred == k))[:8]
rows = []
for i in idx:
    actual, on_time = rhythm_counterfactual(model_c, ds2["X"][i], ds2["D"][i], ds2["rr"][i], temperature=T)
    rows.append({"record": int(ds2["rec"][i]), "RR before / median RR": round(float(ds2["rr"][i, 0]), 2),
                 "p(S), real beat": round(float(actual[k]), 2), "p(S), if on time": round(float(on_time[k]), 2)})
pd.DataFrame(rows)
'''),
md('''
## Lab 7. From beats to patient risk: an AI Holter

A real Holter has no annotations. The full pipeline (`cardioscope/holter.py`):

1. automatic QRS detection (XQRS algorithm),
2. classification of every beat, with a confidence,
3. clinical indicators: PVC burden, ventricular runs (non-sustained VT), couplets, bigeminy,
   supraventricular activity (Binici et al., Circulation 2010), pauses, simplified Lown grade.

We analyse four **unseen** patients and compare with the cardiologists. Watch record 219 (atrial fibrillation):
how many beats does the model refuse to decide on?
'''),
code(r'''
reports = {}
for rec in (100, 233, 232, 219):
    sig, fs, samples, symbols = D.load_record(rec)
    res = analyze_ecg(sig, fs, model_c, temperature=T, threshold=0.8)
    ref = compare_with_reference(res, samples, symbols, fs)
    reports[rec] = (res, ref)
    print(f"=== Record {rec} | automatic QRS detection: Se {100 * ref['qrs_se']:.1f} %, PPV {100 * ref['qrs_ppv']:.1f} %")
    print(format_report(res.report), "\n")
'''),
code(r'''
res, ref = reports[233]
first_v = res.positions[res.labels == C.CLASSES.index("V")][0] / res.fs
viz.plot_holter_strip(res, first_v - 3, 10, ref, "Record 233: AI labels (top) vs cardiologists (bottom)")
'''),
code(r'''per_class(ref["y_true"], ref["y_pred"])'''),
code(r'''viz.plot_tachogram(reports[232][0], "Record 232: RR tachogram, pauses appear above the dashed line")'''),
md('''
## Deploy CardioScope

One cell turns the model into a web application. On Colab, open the public `*.gradio.live` link on your phone.
'''),
code(r'''
from app import build_app, launch
demo = build_app("checkpoints/cardionet_context.pt")   # or your own: "checkpoints/my_cardionet_context.pt"
launch(demo, share=IN_COLAB, prevent_thread_lock=True)
'''),
md('''
## Discussion: responsible AI in cardiology (15 min)

1. **Data.** 47 patients, Boston, 1975-1979, 30-minute recordings, lead MLII. Would the model work on a Tunisian
   population? On a smartwatch (lead I)? On a 12-lead ECG? How would you check?
2. **Blind spots.** No atrial fibrillation class, paced beats excluded, noisy segments. What happens with record 219?
3. **Regulation.** Software that detects arrhythmias for diagnosis is a medical device (EU MDR) and a high-risk AI system
   under the EU AI Act. In Tunisia, health data processing falls under organic law 2004-63 on personal data protection
   and the supervision of the INPDP.
4. **Human in the loop.** Who is responsible for a missed ventricular tachycardia? Who chooses the referral threshold?
5. **Next steps.** 12-lead deep learning (PTB-XL), external validation on other databases, prospective study.

*CardioScope is teaching material, not a medical device.*
'''),
]

notebook(p1, "Part1_ECG_to_Diagnosis.ipynb")
notebook(p2, "Part2_Model_to_Clinic.ipynb")
