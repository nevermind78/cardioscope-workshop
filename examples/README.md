# Example ECG files

60-second excerpts (lead MLII, millivolts, 360 Hz) of three records of the
MIT-BIH Arrhythmia Database that the model never saw during training:

| file | content |
|---|---|
| `mitdb_100_60s.csv` | normal sinus rhythm |
| `mitdb_233_60s.csv` | frequent premature ventricular contractions |
| `mitdb_232_60s.csv` | premature atrial contractions and a long pause |

Format understood by CardioScope: one sample per line (header lines are
ignored; a `Sample Rate,<n> hertz` line sets the sampling rate).

Source: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Eng in Med and Biol 20(3):45-50 (2001). Distributed by PhysioNet under
the Open Data Commons Attribution License v1.0.
