"""Instructor script (run it once before the workshop, on a good connection).

    python scripts/prepare_data.py                 # PhysioNet -> data/mitdb + cache
    python scripts/prepare_data.py --zip mitdb.zip # offline: official PhysioNet ZIP
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardioscope import config as C, data as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--zip", help="path to mit-bih-arrhythmia-database-1.0.0.zip (plan B)")
ap.add_argument("--data-dir", default=str(D.DEFAULT_DATA_DIR))
args = ap.parse_args()
if args.zip:
    D.extract_mitdb_zip(args.zip, args.data_dir)
else:
    D.download_mitdb(args.data_dir)
data = D.build_dataset(args.data_dir, force=True)
for name, recs in (("DS1 (train)", C.DS1), ("DS2 (test)", C.DS2)):
    print(name, D.class_counts(D.by_records(data, recs)).to_dict())
