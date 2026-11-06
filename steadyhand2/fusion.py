"""Agree-or-hold: a hand that moves only when muscles and camera agree.

Two classifiers look at the same moment:

  EMG     p_emg over  [rest, grip_1, ..., grip_K]   "is the user trying to grab, and how?"
  vision  p_vis over  [grip_1, ..., grip_K]         "what grip does this object afford?"

The gate acts with grip g only if

  1. EMG's top class is g (not rest),
  2. vision's top class is also g, and
  3. the fused confidence  c = (1 - p_emg[rest]) * q[g]  is above a cutoff,
     where q is the normalized product of the two grip distributions
     q[g] = p_emg[g] * p_vis[g] / sum_h p_emg[h] * p_vis[h].

Otherwise the hand holds still. A *misfire* is an action when the user did
not intend to grab (true class rest) or with the wrong grip.

The cutoff is chosen with SteadyHand's tools on the fused confidence, so the
promise ("misfire on at most 5% of decisions") carries over:
ConformalCutoff on average over new users, AdaptiveCutoff for each user on
any sequence. What is new is the question the paper asks: when one signal
degrades (electrode shift, bad lighting, a strange object), does the
combined system keep the promise, and how much does it still move?

`mode` gives the baselines with the same interface:
  "emg"      EMG alone (SteadyHand 1)
  "vision"   camera decides the grip, EMG only says "grab now"
             (acts if EMG top != rest; conf = (1-p_rest) * p_vis[g])
  "product"  product of experts without the agreement check
  "agree"    the full rule above (default)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REST = 0


@dataclass
class Decisions:
    pred: np.ndarray      # (n,) chosen class in EMG's label space (0 = rest / hold)
    conf: np.ndarray      # (n,) confidence used against the cutoff
    eligible: np.ndarray  # (n,) bool, the rule allows acting at all (before the cutoff)


def fuse(p_emg, p_vis, mode: str = "agree") -> Decisions:
    """p_emg: (n, K+1) with column 0 = rest. p_vis: (n, K). Rows sum to 1."""
    p_emg = np.asarray(p_emg, float)
    p_vis = np.asarray(p_vis, float)
    n, k1 = p_emg.shape
    if p_vis.shape != (n, k1 - 1):
        raise ValueError(f"p_vis must be (n, {k1 - 1}), got {p_vis.shape}")
    p_rest = p_emg[:, REST]
    g_emg = p_emg.argmax(1)                         # 0 = rest
    grips_e = p_emg[:, 1:]
    go = 1.0 - p_rest

    if mode == "emg":
        pred = g_emg
        conf = p_emg.max(1)
        eligible = g_emg != REST
    elif mode == "vision":
        g = p_vis.argmax(1) + 1
        pred = np.where(g_emg != REST, g, REST)
        conf = go * p_vis.max(1)
        eligible = g_emg != REST
    elif mode in ("product", "agree"):
        prod = grips_e * p_vis
        q = prod / np.clip(prod.sum(1, keepdims=True), 1e-12, None)
        g = q.argmax(1) + 1
        conf = go * q.max(1)
        if mode == "product":
            eligible = go > p_rest
        else:
            eligible = (g_emg == g) & (p_vis.argmax(1) + 1 == g)
        pred = np.where(eligible, g, REST)
    else:
        raise ValueError(mode)
    conf = np.where(eligible, conf, 0.0)
    return Decisions(pred=pred, conf=conf, eligible=eligible)


def act(dec: Decisions, cutoff: float) -> np.ndarray:
    """(n,) class to execute, 0 = hold still."""
    return np.where(dec.eligible & (dec.conf > cutoff), dec.pred, REST)


def misfires(executed, y) -> np.ndarray:
    """Acted (executed != rest) and not what the user wanted."""
    executed, y = np.asarray(executed), np.asarray(y)
    return (executed != REST) & (executed != y)


def summarize(executed, y) -> dict:
    executed, y = np.asarray(executed), np.asarray(y)
    want = y != REST
    m = misfires(executed, y)
    return {
        "misfire_rate": float(m.mean()),
        "act_rate": float((executed != REST).mean()),
        "correct_grab_rate": float(((executed == y) & want).sum() / max(want.sum(), 1)),
        "false_grab_rate": float(((executed != REST) & ~want).sum() / max((~want).sum(), 1)),
    }


def wrong_for_cutoff(dec: Decisions, y) -> np.ndarray:
    """What SteadyHand's cutoff tools call `wrong`: the decision would misfire
    if it acted. Ineligible decisions have conf 0 and never act."""
    return dec.eligible & (dec.pred != np.asarray(y))
