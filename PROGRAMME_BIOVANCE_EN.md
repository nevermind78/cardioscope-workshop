# BIOVANCE 2026 workshop: Deep Learning for Cardiovascular Risk Detection

**From raw ECG to an AI Holter assistant**

* **Speaker:** Abdallah Khemais, faculty member, ISITCOM, University of Sousse
* **Date and venue:** Saturday, September 26, 2026, École Polytechnique de Sousse
* **Format:** hands-on workshop, 3.5 hours in two sessions, Python notebooks on Google Colab (no installation)

## Abstract

Cardiovascular diseases are the leading cause of death worldwide, and the electrocardiogram is the most widely used
cardiac test. In this hands-on workshop, participants build **CardioScope**, an AI Holter assistant that detects every
heartbeat of an ECG, classifies it (normal, supraventricular, ventricular, fusion) and turns the result into a
patient-level arrhythmic risk profile. Working with the MIT-BIH Arrhythmia Database and PyTorch, they train 1D
convolutional networks, discover why many "99 % accurate" models fail on new patients, and teach the network to reason
like a cardiologist. The afternoon session covers calibration, uncertainty-based triage, explainability and deployment
as a web application, and closes with a debate on responsible AI in cardiology.

## Learning outcomes

* Preprocess and segment ECG signals from a reference clinical database
* Train 1D CNNs and evaluate them with a patient-wise protocol and clinical metrics (sensitivity, PPV)
* Give a deep network the clinical context a cardiologist uses (rhythm, the patient's usual beat)
* Calibrate probabilities, refer uncertain cases to a human, explain decisions with Grad-CAM
* Deploy a model as a web application and discuss its clinical, regulatory and ethical limits

## Schedule

| Time | Part 1: from raw ECG to beat-by-beat diagnosis |
|---|---|
| 9:00 to 9:15 | Opening: the ECG as a digital biomarker, "human vs AI" quiz |
| 9:15 to 9:35 | Lab 1: exploring the MIT-BIH Arrhythmia Database |
| 9:35 to 10:00 | Lab 2: a first 1D convolutional network |
| 10:00 to 10:20 | Lab 3: the data-leakage trap and clinical metrics |
| 10:20 to 10:50 | Lab 4: thinking like a cardiologist (rhythm and patient context) |
| 10:50 to 11:00 | Wrap-up and challenge |

| Time | Part 2: from model to clinical tool |
|---|---|
| 14:00 to 14:10 | Recap |
| 14:10 to 14:30 | Lab 5: an AI that knows when it doesn't know (calibration, triage) |
| 14:30 to 14:45 | Lab 6: opening the black box (Grad-CAM, counterfactuals) |
| 14:45 to 15:10 | Lab 7: from beats to patient risk, and live deployment of the app |
| 15:10 to 15:25 | Debate: responsible AI in cardiology |
| 15:25 to 15:30 | Closing |

## Participants should bring

A charged laptop, a Google account and a smartphone. Prerequisite: basic Python.
