import numpy as np

from wf2_drift_validity.recovery_metrics import (
    first_sustained,
    ranking_recovery,
    accuracy_recovery,
)


def test_first_sustained():
    condition = [False, True, True, False, True, True, True]
    assert first_sustained(condition, 3) == 4


def test_ranking_immediate_recovery():
    t = np.arange(50, 501, 50)
    precision = np.ones(len(t))

    result = ranking_recovery(t, precision, drift_end=200)

    assert result["recovered"]
    assert result["recovery"] == 50


def test_ranking_delayed_recovery():
    t = np.arange(50, 601, 50)

    precision = [
        1, 1, 1, 1,
        0, 0.33, 0.67, 1,
        1, 1, 1, 1
    ]

    result = ranking_recovery(t, precision, drift_end=200)

    assert result["recovered"]
    assert result["recovery"] == 200


def test_ranking_never_recovers():
    t = np.arange(50, 501, 50)
    precision = np.full(len(t), 0.67)

    result = ranking_recovery(t, precision, drift_end=200)

    assert not result["recovered"]
    assert np.isnan(result["recovery"])
    assert result["censored_time"] == 300


def test_temporary_recovery():
    t = np.arange(50, 501, 50)

    precision = [
        1, 1, 1, 1,
        1, 1, 0, 1, 1, 1
    ]

    result = ranking_recovery(t, precision, drift_end=200)

    assert result["recovered"]
    assert result["recovery"] == 200


def test_gradual_drift():
    t = np.arange(50, 1001, 50)
    precision = np.ones(len(t))

    result = ranking_recovery(t, precision, drift_end=600)

    assert result["recovered"]
    assert result["recovery"] == 50


def test_accuracy_recovery():
    t = np.arange(50, 2001, 50)

    accuracy = np.where(
        t <= 1000,
        0.85,
        np.where(t <= 1250, 0.60, 0.84)
    )

    result = accuracy_recovery(t, accuracy, drift_start=1000)

    assert result["recovered"]
    assert result["recovery"] == 300
    assert np.isclose(result["accuracy_drop"], 0.25)


def test_accuracy_never_recovers():
    t = np.arange(50, 2001, 50)
    accuracy = np.where(t <= 1000, 0.85, 0.60)

    result = accuracy_recovery(t, accuracy, drift_start=1000)

    assert not result["recovered"]
    assert np.isnan(result["recovery"])
    assert result["censored_time"] == 1000