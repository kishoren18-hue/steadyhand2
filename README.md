# SteadyHand 2: a hand that looks before it grabs

A wrist camera guesses the grip the object affords. EMG says *when* to grab
and *which* grip. The hand moves only when both agree with enough confidence;
otherwise it holds still. The misfire promise ("at most 5% of decisions")
comes from [SteadyHand](https://github.com/kishoren18-hue/steadyhand)'s
conformal and adaptive cutoffs, applied to the fused confidence.

**The research question:** when one signal degrades (electrode shift, dim
light, an unfamiliar object), does the combined system keep the promise, and
how much can the user still grab? Vision+EMG fusion is an active field. This
project's angle is the stated misfire guarantee and the degradation analysis.

> **Status: code + simulation only.** `results/sim/` comes from simulated
> users, not people. Don't report it.

## Gate modes (same interface, so baselines are one flag away)

| mode | acts when | confidence |
|---|---|---|
| `emg` | EMG top class isn't rest | EMG max prob (SteadyHand 1) |
| `vision` | EMG says "grab" | (1 − p_rest) × vision max |
| `product` | grab more likely than rest | (1 − p_rest) × product-of-experts max |
| `agree` | EMG top == vision top == product top | (1 − p_rest) × product-of-experts max |

## What simulation suggests (not data)

20 clean calibration users, 15 test users per condition, 5% target, conformal
cutoff calibrated on clean data:

| condition | EMG only: mean misfire | agree gate: mean misfire | agree gate: correct grabs |
|---|---|---|---|
| clean | 0.3% | 0.3% | 84% |
| EMG electrode shift (20% confidently wrong) | **17.3%** | **3.2%** | 69% |
| both degraded (shift + dim light) | 17.5% | 3.6% | 48% |

Agreement acts as a structural guard: one signal going confidently wrong
can't move the hand alone. Real data has to confirm it. Full table:
`results/sim/sim_output.txt`.

## Pieces

* `fusion.py`: the gate and misfire accounting
* `calibrate.py`: SteadyHand's `crc_threshold` and `AdaptiveCutoff` on the fused confidence
* `vision.py`: frozen ResNet-18 or V-JEPA 2 features + calibrated logistic-regression grip head
* `record.py`: synced EMG (any LibEMG 2.0.3 streamer) + wrist video with on-screen cues. Reads LibEMG's newest-first buffer by matching rows, because streamers disagree on what `emg_count` counts.
* `hand.py`: the SO-101 gripper as the hand for a live demo (grip → closure)
* `simulate.py`, `scripts/sim_experiment.py`: simulated users and the degradation sweep

```bash
pip install -e ../steadyhand -e ".[test]"     # Python 3.10–3.12 (LibEMG needs numpy<2)
pytest -q
python scripts/sim_experiment.py
python -m steadyhand2.record --device oymotion --camera 0 --out data/s01 --reps 5
```

**Recording other people** needs written consent and usually an ethics
review (IRB) before publishing. Ask your Georgia Tech BME contact how their
lab handles it. Start with yourself.
