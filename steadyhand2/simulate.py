"""Simulated users for testing the fusion gate before any wrist video exists.

NOT DATA. Each simulated person has an EMG skill level and sees objects with
some visual clarity. Degradations model what the paper will test for real:

  emg_shift    electrode shift: EMG becomes confidently wrong on a share of
               decisions (SteadyHand 1 measured 46% misfires at LibEMG's 0.90
               cutoff after a 45-degree shift, so this failure mode is real)
  emg_noise    EMG gets less certain overall
  vis_dark     vision gets less certain (dim light, motion blur)
  vis_novel    vision is confidently wrong on unfamiliar objects

Classes: 0 = rest, 1..K = grips. The user wants rest on `p_rest` of
decisions; on the rest they want the grip the object affords.
"""
from __future__ import annotations

import numpy as np


def _softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def person(n=2000, K=4, emg_skill=3.0, vis_skill=3.0, p_rest=0.6, emg_shift=0.0, emg_noise=0.0,
           vis_dark=0.0, vis_novel=0.0, rng=None):
    r = rng or np.random.default_rng()
    # intent comes in runs (a grab lasts a while), like real EMG segments
    y = np.zeros(n, int)
    t = 0
    while t < n:
        L = int(r.integers(10, 40))
        y[t:t + L] = 0 if r.random() < p_rest else int(r.integers(1, K + 1))
        t += L
    afford = np.where(y > 0, y, r.integers(1, K + 1, n))       # grip the object in view affords

    # EMG over [rest, grips]
    s_e = max(emg_skill * (1 - emg_noise), 0.1)
    le = r.normal(0, 1, (n, K + 1))
    le[np.arange(n), y] += s_e
    flip = r.random(n) < emg_shift                               # confidently wrong
    wrong = (y + r.integers(1, K + 1, n)) % (K + 1)
    le[flip] = r.normal(0, 1, (flip.sum(), K + 1))
    le[np.flatnonzero(flip), wrong[flip]] += emg_skill + 1.5
    p_emg = _softmax(le)

    # vision over grips (what the object affords)
    s_v = max(vis_skill * (1 - vis_dark), 0.1)
    lv = r.normal(0, 1, (n, K))
    lv[np.arange(n), afford - 1] += s_v
    nov = r.random(n) < vis_novel
    wv = (afford - 1 + r.integers(1, K, n)) % K
    lv[nov] = r.normal(0, 1, (nov.sum(), K))
    lv[np.flatnonzero(nov), wv[nov]] += vis_skill + 1.0
    p_vis = _softmax(lv)
    return {"p_emg": p_emg, "p_vis": p_vis, "y": y}


def population(n_people=20, rng=None, **degrade):
    r = rng or np.random.default_rng(0)
    return [person(emg_skill=r.uniform(2.0, 4.5), vis_skill=r.uniform(2.5, 4.5), rng=r, **degrade)
            for _ in range(n_people)]
