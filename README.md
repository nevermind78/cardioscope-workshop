# CardioScope: Deep Learning for Cardiovascular Risk Detection

Hands-on workshop material, **BIOVANCE 2026** (IEEE EMBS EPS SBC and IEEE EMBS IIT SBC),
École Polytechnique de Sousse, September 26, 2026. Instructor: Abdallah Khemais.

Participants build **CardioScope**, an AI Holter assistant that reads a single-lead ECG like a cardiologist:
it detects every heartbeat, classifies it (AAMI classes N, S, V, F), turns the labels into a patient-level
arrhythmic risk profile, and is deployed as a web app they open on their phone.

> Educational material. CardioScope is **not a medical device** and must never be used for diagnosis.

## Program

| Part | Time | Labs |
|---|---|---|
| 1. From raw ECG to beat-by-beat diagnosis | 9:00 to 11:00 | 1. Explore MIT-BIH.<br>2. A first 1D CNN.<br>3. The leakage trap.<br>4. Think like a cardiologist (rhythm + patient's dominant beat) |
| 2. From model to clinical tool | 14:00 to 15:30 | 5. Calibration and triage.<br>6. Grad-CAM and counterfactuals.<br>7. AI Holter report + deployment. Responsible AI debate |

## Quick start

### Google Colab (participants)

1. Open `Part1_ECG_to_Diagnosis.ipynb` on GitHub and click the **Open in Colab** badge at the top
   (*Runtime > Change runtime type > T4 GPU*, optional), then run the cells in order.
   The setup cell clones this repository and installs `wfdb` and `gradio`; the first lab downloads MIT-BIH
   from PhysioNet (~90 MB, 1 to 2 min).
2. Afternoon: same for `Part2_Model_to_Clinic.ipynb`. If Colab has reset the machine, the notebook reloads
   the pretrained models of `checkpoints/`.


### Local machine (instructor, CUDA GPU)

```bash
# 1. PyTorch with CUDA: follow https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python scripts/prepare_data.py            # PhysioNet -> data/mitdb + beat cache (plan B: --zip <PhysioNet ZIP>)
python scripts/train.py --model context   # retrain a model; also: morph, rhythm
python app.py                             # http://127.0.0.1:7860   (--share for a public link)
jupyter lab                               # the two notebooks run locally too
```

## Repository layout

```
cardioscope/            teaching package
  config.py             sampling rates, AAMI classes, inter-patient split (de Chazal 2004)
  data.py               PhysioNet download, filtering, beat windows, RR features, patient context
  models.py             CardioNet: compact 1D ResNet + rhythm branch + dominant-beat channel
  training.py           training loop, class weights, jitter augmentation, checkpoints
  metrics.py            Se / PPV per class, macro-F1, ECE, temperature scaling, triage
  explain.py            Grad-CAM 1D, rhythm counterfactual
  holter.py             QRS detection (XQRS), whole-recording analysis, risk report, comparison with cardiologists
  viz.py                ECG-paper strips, tachogram, confusion matrices, reliability diagrams
app.py                  Gradio app "CardioScope"
Part1_ECG_to_Diagnosis.ipynb, Part2_Model_to_Clinic.ipynb
checkpoints/            pretrained models (morph, rhythm, context), with calibrated temperature
examples/               three 60 s ECG files of unseen patients for the app
scripts/                prepare_data.py, train.py
tools/build_notebooks.py  regenerates the notebooks
PROGRAMME_BIOVANCE_EN.md  program for the organizers
```

## Results of the pretrained models

Trained on the 22 DS1 patients, 12 epochs, seed 0, CPU. "Same patients" = 10 % of DS1 beats held out at random;
"new patients" = the 22 DS2 patients never seen in training.

| Model | Accuracy, same patients | Accuracy, new patients | Macro-F1, new patients | V: Se / PPV | S: Se / PPV |
|---|---|---|---|---|---|
| 1. Morphology | 98.9 % | 76.1 % | 40.5 % | 93.0 / 46.3 % | 22.1 / 9.5 % |
| 2. + rhythm (RR) | 99.1 % | 78.2 % | 44.6 % | 96.0 / 58.5 % | 17.2 / 17.3 % |
| 3. + patient's dominant beat | 99.3 % | 93.0 % | 57.0 % | 95.4 / 90.7 % | 24.7 / 27.0 % |

AI Holter on unseen patients, automatic QRS detection (no annotation used), model 3:

| Record | QRS detection Se / PPV | Finding |
|---|---|---|
| 100 | 100 / 100 % | 33 of 33 premature atrial beats found |
| 233 | 99.8 / 100 % | PVCs: Se 97.7 %, PPV 99.1 %; burden 26.5 %, 5 ventricular runs: high risk |
| 214 | 99.9 / 100 % | LBBB patient, PVCs: Se 98.4 %, PPV 95.4 % |
| 232 | 100 / 99.9 % | 5.9 s pause detected; premature atrial beats mostly missed (Se 13 %) |
| 219 | 99.7 / 100 % | atrial fibrillation: false S beats, but 73 % of beats flagged for human review |

Supraventricular beats in unseen patients remain hard: this is discussed openly in the workshop.

## Data and references

* MIT-BIH Arrhythmia Database, PhysioNet, Open Data Commons Attribution License v1.0.
  Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng Med Biol* 20(3):45-50, 2001.
* Pollard T, Moody BE, Lehman L, et al. PhysioNet as a global platform for biomedical research. *Nature Health*, 2026.
* de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE TBME* 51(7), 2004.
* ANSI/AAMI EC57, Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms.
* Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML* 2017.
* Selvaraju RR et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. *ICCV* 2017.
* Binici Z et al. Excess supraventricular ectopic activity and increased risk of atrial fibrillation and stroke. *Circulation* 121, 2010.
* Hannun AY et al. Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine* 25, 2019.

Code released under the MIT License (see `LICENSE`).
