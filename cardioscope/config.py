"""Global configuration of the CardioScope workshop.

Everything that defines *the experiment* lives here, so that the notebooks,
the training script and the Gradio app all share exactly the same settings.
"""

# --------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------
MITDB_FS = 360            # native sampling rate of the MIT-BIH Arrhythmia DB (Hz)
FS = 180                  # working sampling rate (Hz): plenty for beat morphology
BANDPASS = (0.5, 40.0)    # Hz: removes baseline wander (<0.5 Hz) and EMG / mains noise

PRE_R_S, POST_R_S = 0.25, 0.46          # beat window around the R peak (seconds)
PRE_R = int(round(PRE_R_S * FS))        # 45 samples
POST_R = int(round(POST_R_S * FS))      # 83 samples
WIN = PRE_R + POST_R                    # 128 samples = model input length
JITTER = 4                              # +/- samples of random shift (augmentation)
STORE = WIN + 2 * JITTER                # 136 samples stored per beat

# --------------------------------------------------------------------------
# Labels: ANSI/AAMI EC57 heartbeat classes
# --------------------------------------------------------------------------
CLASSES = ["N", "S", "V", "F"]
CLASS_NAMES = {
    "N": "Normal / bundle branch block",
    "S": "Supraventricular ectopic (SVEB)",
    "V": "Ventricular ectopic (VEB / PVC)",
    "F": "Fusion (ventricular + normal)",
}
CLASS_COLORS = {"N": "#4A6FA5", "S": "#D98E04", "V": "#C0392B", "F": "#7D3C98"}

# MIT-BIH beat symbol -> AAMI class.  Q-class beats (paced '/', 'f', 'Q') are
# excluded from training/evaluation, as in most inter-patient studies.
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}
# Every annotation symbol that marks a heartbeat (used to compute RR intervals,
# including beats we do not classify, so the rhythm context stays correct).
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")

# --------------------------------------------------------------------------
# Inter-patient protocol of de Chazal et al., IEEE TBME 2004
# (the 4 paced records 102, 104, 107, 217 are excluded)
# --------------------------------------------------------------------------
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]   # training patients
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]   # unseen test patients
PACED = [102, 104, 107, 217]
ALL_RECORDS = sorted(DS1 + DS2)

# --------------------------------------------------------------------------
# Rhythm (RR) features fed to the second branch of the network
# --------------------------------------------------------------------------
RR_FEATURES = [
    "pre_RR / median_RR",      # how early is this beat compared to the patient's rhythm?
    "post_RR / median_RR",     # is it followed by a (compensatory) pause?
    "local_RR / median_RR",    # current local rhythm (last 10 beats)
    "pre_RR / local_RR",       # prematurity relative to the *local* rhythm
    "post_RR / pre_RR",        # classic compensatory-pause ratio
]
N_RR = len(RR_FEATURES)

# Part 2 — calibration must be learned on *new* patients too:
CAL_RECORDS = DS2[::2]     # calibration cohort (11 unseen patients)
EVAL_RECORDS = DS2[1::2]   # evaluation cohort (11 other unseen patients)
