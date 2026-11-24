import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe, FeatureSeparabilityBuffer
from run_relevance import run_one
from wf3_relevance.collect import FORGETTING, LocalWindow, RecursiveSeparability, RelevanceCollector


ROOT = Path(__file__).resolve().parents[1]


def test_recursive_matches_weighted_batch():
    rng = np.random.default_rng(19)
    x = rng.random((730, 5))
    y = rng.integers(0, 2, len(x))
    y[-1] = 2  # The singleton still contributes to the overall mean.
    recursive = RecursiveSeparability(3, x.shape[1])
    batch = FeatureSeparabilityBuffer(x.shape[1], len(x))
    for row, label in zip(x, y):
        recursive.add(row, label)
        batch.add(row, label)

    results = recursive.weights()
    np.testing.assert_allclose(results[-1], batch.compute_feature_weights(), atol=1e-12)
    for decay, got in zip(FORGETTING[:-1], results):
        mass = decay ** np.arange(len(x) - 1, -1, -1)
        overall = np.average(x, axis=0, weights=mass)
        ratio = np.zeros(x.shape[1])
        for label in (0, 1):
            rows, weights = x[y == label], mass[y == label]
            center = np.average(rows, axis=0, weights=weights)
            scatter = (weights[:, None] * (rows - center)**2).sum(axis=0)
            ratio += weights.sum() * (center - overall)**2 / (scatter + 1e-6)
        np.testing.assert_allclose(got, 0.2 + 0.8 * ratio / ratio.max(), atol=1e-12)


def test_r3_window_can_become_invalid_again():
    local = LocalWindow(2)
    for t in range(1, 20):
        local.add([0.0, 0.0], 0, t)
    assert np.isnan(local.snapshot()[3]).all()
    local.add([0.9, 0.0], 1, 20)
    assert np.isfinite(local.snapshot()[3]).all()
    for t in range(21, 221):
        local.add([0.3, 0.2], 0, t)
    size, first, last, weights = local.snapshot()
    assert (size, first, last) == (200, 21, 220)
    assert np.isnan(weights).all()


@pytest.mark.parametrize("max_rules", [5, None])
def test_collection_preserves_training_and_reproduces_r1(max_rules, monkeypatch, tmp_path):
    cfg = json.loads((ROOT / "configs/protocol/s1_gauss_abrupt.json").read_text())
    cfg["stream"].update(n=1400, drift_at=700)
    config = cfg["variants"]["ref" if max_rules == 5 else "open"]
    traces = []
    fit = EvolvingNeuroFuzzyUnimSafe.partial_fit

    def observe_fit(model, *args):
        fit(model, *args)
        traces.append([(r.center.copy(), r.w.copy(), r.support) for r in model.rules])

    monkeypatch.setattr(EvolvingNeuroFuzzyUnimSafe, "partial_fit", observe_fit)
    plain = run_one(cfg, 0, model_cfg=config)
    collected = run_one(cfg, 0, model_cfg=config, wf3=True)
    np.testing.assert_array_equal(plain["y_pred"], collected["y_pred"])
    np.testing.assert_array_equal(plain["n_rules"], collected["n_rules"])
    for a_step, b_step in zip(traces[:1400], traces[1400:], strict=True):
        for (a_center, a_w, a_support), (b_center, b_w, b_support) in zip(a_step, b_step, strict=True):
            np.testing.assert_array_equal(a_center, b_center)
            np.testing.assert_array_equal(a_w, b_w)
            assert a_support == b_support

    collected["wf3"].save(tmp_path)
    with (tmp_path / "wf3_checkpoints.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 28
    for row, global_weights in zip(rows, collected["logger"].arrays()["G"], strict=True):
        np.testing.assert_array_equal([float(row[f"window_200_x{i}"]) for i in range(10)], global_weights)
        assert 0 <= float(row["r3_coverage_pct"]) <= 100


def test_r3_tracks_rule_winners_through_removal_and_merge():
    model = EvolvingNeuroFuzzyUnimSafe(2, 2, min_rules_for_merge=1, min_support_for_merge=1)
    collector = RelevanceCollector(model, None)
    model._create_rule(np.array([0.0, 0.0]), 0, 1.0)
    model._create_rule(np.array([0.1, 0.1]), 1, 1.0)
    survivor, removed = model.rules
    collector.after_fit(0, False)
    collector.observe_prediction(np.array([0.0, 0.0]), 0, 1)
    collector.observe_prediction(np.array([0.1, 0.1]), 1, 2)
    survivor_id, removed_id = collector.tracker.uid(survivor), collector.tracker.uid(removed)
    assert set(collector.local) == {survivor_id, removed_id}

    model._maybe_merge_last_rule()
    collector.after_fit(2, True)
    assert len(model.rules) == 1 and model.rules[0] is survivor
    assert list(collector.local) == [survivor_id]
    assert list(collector.local[survivor_id].times) == [2]
    assert collector.rule_rows[0][:5] == [3, survivor_id, 1, 2, 2]
    assert np.isnan(collector.rule_rows[0][5:]).all()
