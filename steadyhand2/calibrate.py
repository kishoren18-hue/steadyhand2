"""SteadyHand's cutoffs, applied to the fused confidence.

Both functions only need (conf, wrong) per decision, which `fusion` provides,
so the guarantees are exactly SteadyHand's:

  conformal_cutoff  expected misfire rate of a new user <= target, if the new
                    user is exchangeable with the calibration users
  run_adaptive      each user's long-run misfire rate <= target + a vanishing
                    term, on any sequence, given feedback on decisions
"""
from __future__ import annotations

import numpy as np

from steadyhand.adaptive import simulate
from steadyhand.cutoffs import crc_threshold
from steadyhand.risk import candidate_thresholds, misfire_curve

from .fusion import fuse, wrong_for_cutoff


def conformal_cutoff(people, target: float, mode: str = "agree") -> float:
    """people: list of dicts with 'p_emg', 'p_vis', 'y' from models that never
    trained on that person. Returns a cutoff on the fused confidence."""
    decs = [fuse(p["p_emg"], p["p_vis"], mode) for p in people]
    grid = candidate_thresholds(np.concatenate([d.conf for d in decs]))
    curves = [misfire_curve(d.conf, wrong_for_cutoff(d, p["y"]), grid) for d, p in zip(decs, people)]
    return crc_threshold(grid, curves, target)


def run_adaptive(person, target: float, start: float, mode: str = "agree", eta: float = 0.5, feedback=None):
    """Replay one person's decisions through SteadyHand's AdaptiveCutoff.
    Returns (executed classes, cutoff trace on the confidence scale)."""
    d = fuse(person["p_emg"], person["p_vis"], mode)
    wrong = wrong_for_cutoff(d, person["y"])
    acted, tau = simulate(d.conf, wrong, target, start, eta=eta, feedback=feedback)
    executed = np.where(acted & d.eligible, d.pred, 0)
    return executed, 1 - 10 ** (-np.minimum(tau, 6.0))
