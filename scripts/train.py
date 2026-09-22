"""Train one CardioNet variant on DS1 and evaluate it on the 22 unseen patients of DS2.

    python scripts/train.py --model context --epochs 12      # ~3.5 min on 1 CPU core, much faster on GPU
    python scripts/train.py --model morph --out checkpoints/cardionet_morph.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
from cardioscope import config as C, data as D  # noqa: E402
from cardioscope.metrics import fit_temperature, per_class, summary  # noqa: E402
from cardioscope.models import CardioNet, count_parameters  # noqa: E402
from cardioscope.training import get_device, predict_logits, save_checkpoint, train_model  # noqa: E402

VARIANTS = {"morph": {}, "rhythm": {"use_rr": True}, "context": {"use_rr": True, "use_context": True}}
ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=VARIANTS, default="context")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--weight-power", type=float, default=0.5)
ap.add_argument("--width", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()

device = get_device()
data = D.build_dataset()
ds1, ds2 = D.by_records(data, C.DS1), D.by_records(data, C.DS2)
train, val = D.holdout_split(ds1, 0.1, seed=0)
cal = D.by_records(data, C.CAL_RECORDS)
model = CardioNet(width=args.width, **VARIANTS[args.model])
print(f"{args.model}: {count_parameters(model):,} parameters")
train_model(model, train, val, epochs=args.epochs, lr=args.lr, weight_power=args.weight_power,
            seed=args.seed, device=device)
pred_val = predict_logits(model, val).argmax(1)
pred_test = predict_logits(model, ds2).argmax(1)
print(pd.DataFrame({"same patients (val)": summary(val["y"], pred_val),
                    "new patients (DS2)": summary(ds2["y"], pred_test)}).T.to_string())
print(per_class(ds2["y"], pred_test).to_string())
T = fit_temperature(predict_logits(model, cal), cal["y"])
out = args.out or f"checkpoints/my_cardionet_{args.model}.pt"
metrics = {"ds2_" + k.split(" (")[0].lower().replace(" ", "_").replace("-", "_"): v
           for k, v in summary(ds2["y"], pred_test).items()}
save_checkpoint(model, out, temperature=T, epochs=args.epochs, seed=args.seed, **metrics)
print(f"saved {out} (temperature {T:.2f})")
