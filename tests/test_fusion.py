import numpy as np
import pytest

from steadyhand2.calibrate import conformal_cutoff, run_adaptive
from steadyhand2.fusion import act, fuse, misfires, summarize
from steadyhand2.record import new_rows
from steadyhand2.simulate import person, population


def test_agree_requires_both():
    p_emg = np.array([[0.1, 0.8, 0.05, 0.05],     # EMG says grip 1
                      [0.1, 0.8, 0.05, 0.05],
                      [0.9, 0.05, 0.03, 0.02]])   # EMG says rest
    p_vis = np.array([[0.9, 0.05, 0.05],          # vision agrees: grip 1
                      [0.05, 0.9, 0.05],          # vision says grip 2
                      [0.9, 0.05, 0.05]])
    d = fuse(p_emg, p_vis, "agree")
    assert d.eligible.tolist() == [True, False, False]
    assert act(d, 0.0).tolist() == [1, 0, 0]
    assert d.conf[0] == pytest.approx(0.9 * (0.8 * 0.9) / (0.8 * 0.9 + 0.05 * 0.05 + 0.05 * 0.05))


def test_modes_shapes_and_bounds():
    r = np.random.default_rng(0)
    pp = person(n=500, rng=r)
    for m in ("emg", "vision", "product", "agree"):
        d = fuse(pp["p_emg"], pp["p_vis"], m)
        assert d.conf.shape == (500,) and (d.conf >= 0).all() and (d.conf <= 1).all()
        ex = act(d, 1.0)
        assert (ex == 0).all()                        # cutoff 1 never acts


def test_misfire_definition():
    assert misfires([0, 1, 2, 1], [0, 1, 1, 0]).tolist() == [False, False, True, True]
    s = summarize([0, 1, 2, 1], [0, 1, 1, 0])
    assert s["misfire_rate"] == 0.5 and s["false_grab_rate"] == 0.5


def test_conformal_holds_on_exchangeable_users():
    r = np.random.default_rng(1)
    cal = population(20, rng=r)
    test = population(40, rng=r)
    c = conformal_cutoff(cal, 0.05, "agree")
    m = np.mean([misfires(act(fuse(p["p_emg"], p["p_vis"], "agree"), c), p["y"]).mean() for p in test])
    assert m <= 0.05 + 0.01


def test_adaptive_holds_each_user_under_shift():
    r = np.random.default_rng(2)
    for _ in range(5):
        p = person(n=3000, emg_shift=0.3, rng=r)
        ex, tau = run_adaptive(p, 0.05, start=0.5, mode="emg")
        assert misfires(ex, p["y"]).mean() <= 0.05 + (6 + 0.5) / (0.5 * 3000) + 1e-9


def test_new_rows_matches_overlap():
    old = np.arange(40, dtype=float).reshape(20, 2)          # newest first
    new = np.vstack([np.full((3, 2), -1.0), old[:-3]])        # 3 new rows pushed on top
    assert new_rows(new, old[:8], count_diff=1) == 3          # count says 1 (MindRove-style); overlap says 3
    assert new_rows(new, None, count_diff=3) == 3
    assert new_rows(new, old[:8], 0) == 3
    assert new_rows(old, old[:8], 0) == 0


def test_grip_classifier_probabilities():
    pytest.importorskip("sklearn")
    from steadyhand2.vision import GripClassifier
    r = np.random.default_rng(3)
    y = r.integers(1, 5, 400)
    X = r.normal(size=(400, 16)) + np.eye(16)[y] * 3
    groups = np.arange(400) // 10
    clf = GripClassifier().fit(X, y, groups=groups)
    p = clf.predict_proba(X, K=4)
    assert p.shape == (400, 4) and np.allclose(p.sum(1), 1)
    assert (p.argmax(1) + 1 == y).mean() > 0.8
