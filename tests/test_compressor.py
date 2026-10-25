import numpy as np
import pytest

from pedalrig.compressor.analysis import fit_static, settle_times

IN = np.arange(-60.0, -5.0, 3.0) + 14.5  # the staircase in dBu


def _hard_knee(in_dbu, thr, ratio, gain):
    over = np.maximum(in_dbu - thr, 0.0)
    return in_dbu + gain - over * (1 - 1 / ratio)


def test_static_threshold_and_ratio_of_hard_knee():
    r = fit_static(IN, _hard_knee(IN, -15.5, 4.0, 5.0))
    assert r["low_gain_db"] == pytest.approx(5.0)
    assert r["low_linear"]
    # 1 dB of reduction is reached 1 / (1 - 1/4) dB above the knee
    assert r["threshold_dbu"] == pytest.approx(-15.5 + 4 / 3, abs=1e-6)
    assert r["ratio"] == pytest.approx(4.0, rel=1e-6)


def test_static_limiter_reports_infinite_ratio():
    out = np.minimum(IN + 10.0, -10.0)
    assert fit_static(IN, out)["ratio"] == float("inf")


def test_static_threshold_above_range_gives_no_threshold_or_ratio():
    r = fit_static(IN, _hard_knee(IN, 20.0, 4.0, 0.0))
    assert r["threshold_dbu"] is None
    assert r["ratio"] is None


def test_static_gain_reduction_runout_excluded_from_ratio():
    # 10:1 above -25 dBu until 20 dB of reduction, then the reduction stops growing (1:1 again)
    gr = np.minimum(np.maximum(IN + 25, 0) * 0.9, 20.0)
    r = fit_static(IN, IN - gr)
    assert r["ratio"] == pytest.approx(10.0, rel=1e-6)
    assert r["gr_limit_db"] == pytest.approx(20.0)
    assert r["runout_dbu"] == pytest.approx(-25 + 20 / 0.9, abs=3.0)


def test_settle_times_of_exponential_gain_change():
    t = np.arange(0, 2.0, 0.001) + 0.0005
    tau = 0.2
    g = -10.0 + 10.0 * np.exp(-t / tau)  # release-like: from 0 dB toward -10 dB
    st = settle_times(t, g, 0.0, -10.0)
    assert st[0.63] == pytest.approx(tau, abs=0.002)
    assert st[0.9] == pytest.approx(tau * np.log(10), abs=0.002)


def test_settle_times_ignore_changes_below_one_db():
    t = np.arange(0, 1.0, 0.001)
    assert settle_times(t, np.zeros_like(t) - 0.5, 0.0, -0.5) == {0.63: None, 0.9: None}
