"""Fusion gate on simulated users: which design keeps the misfire promise
when one signal degrades, and how much does each still let the user grab?

NOT DATA; checks the code and the logic before recording real wrist video.

    PYTHONPATH=.:../steadyhand python scripts/sim_experiment.py
"""
import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from steadyhand2.calibrate import conformal_cutoff, run_adaptive
from steadyhand2.fusion import act, fuse, summarize
from steadyhand2.simulate import population

p = argparse.ArgumentParser()
p.add_argument("--target", type=float, default=0.05)
p.add_argument("--out", default="results/sim")
a = p.parse_args()
Path(a.out).mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
cal = population(20, rng=rng)
conds = {
    "clean": {},
    "emg_shift_20%": {"emg_shift": 0.2},
    "emg_noise": {"emg_noise": 0.5},
    "vis_dark": {"vis_dark": 0.6},
    "vis_novel_20%": {"vis_novel": 0.2},
    "both_degraded": {"emg_shift": 0.2, "vis_dark": 0.6},
}
modes = ["emg", "vision", "product", "agree"]
cut = {m: conformal_cutoff(cal, a.target, m) for m in modes}
print("conformal cutoffs from 20 clean calibration users:", {m: round(c, 4) for m, c in cut.items()})

rows = []
for cname, deg in conds.items():
    test = population(15, rng=np.random.default_rng(zlib.crc32(cname.encode())), **deg)
    for m in modes:
        for method in ("fixed_0.90", "conformal", "adaptive"):
            per = []
            for pp in test:
                if method == "adaptive":
                    ex, _ = run_adaptive(pp, a.target, start=cut[m], mode=m)
                else:
                    c = 0.90 if method == "fixed_0.90" else cut[m]
                    ex = act(fuse(pp["p_emg"], pp["p_vis"], m), c)
                per.append(summarize(ex, pp["y"]))
            df = pd.DataFrame(per)
            rows.append({"condition": cname, "mode": m, "method": method,
                         "misfire_mean": df.misfire_rate.mean(), "misfire_worst": df.misfire_rate.max(),
                         "correct_grab": df.correct_grab_rate.mean(), "false_grab": df.false_grab_rate.mean()})
res = pd.DataFrame(rows)
res.to_csv(f"{a.out}/fusion_sim.csv", index=False)
pd.set_option("display.width", 160)
for val in ("misfire_mean", "misfire_worst", "correct_grab"):
    print(f"\n{val.upper()} (target misfire {a.target:.0%})")
    print(res.pivot_table(index=["condition"], columns=["method", "mode"], values=val).round(3).to_string())
