import numpy as np
import pytest

from pedalrig.analysis import deconvolve, frequency_response, harmonic_irs, hd_vs_freq, ir_peak
from pedalrig.config import FS, POST_S, PRE_S
from pedalrig.signals import SWEEPS, sync_sweep

PRE = int(round(PRE_S * FS))


def _as_capture(y: np.ndarray, delay: int) -> np.ndarray:
    """Place `y` after PRE silence plus `delay` samples, padded like a real capture."""
    out = np.zeros(PRE + len(y) + int(POST_S * FS))
    out[PRE + delay:PRE + delay + len(y)] = y[:len(out) - PRE - delay]
    return out


def test_sweep_harmonics_match_polynomial_closed_form():
    stim = SWEEPS["sweep_m20"]
    p = stim.params
    x = stim.data.astype(np.float64)
    a2, a3 = 0.05, 0.02
    rec = _as_capture(x + a2 * x ** 2 + a3 * x ** 3, 1000)

    h = deconvolve(rec, x, FS, p["f1"], p["f2"])
    i1, pol = ir_peak(h, PRE, PRE + int(0.2 * FS))
    hd = hd_vs_freq(harmonic_irs(h, i1, p["L"], FS), FS, 20000, np.array([1000.0]))

    A = 10 ** (-20 / 20)
    lin = 1 + 3 * a3 * A ** 2 / 4
    assert hd["HD"][2][0] == pytest.approx(20 * np.log10(a2 * A / 2 / lin), abs=0.3)
    assert hd["HD"][3][0] == pytest.approx(20 * np.log10(a3 * A ** 2 / 4 / lin), abs=0.3)
    assert pol == 1


@pytest.mark.parametrize("scale, polarity", [(0.5, 1), (-0.5, -1)])
def test_latency_gain_and_polarity(scale, polarity):
    stim = sync_sweep("lat", 100, 20000, 1.0, -20)
    x = stim.data.astype(np.float64)
    rec = _as_capture(scale * x, 1234)

    h = deconvolve(rec, x, FS, 100, 20000)
    i1, pol = ir_peak(h, PRE, PRE + int(0.2 * FS))
    assert i1 - PRE == pytest.approx(1234, abs=0.1)
    assert pol == polarity

    if scale > 0:
        long = SWEEPS["sweep_m20"]
        xl = long.data.astype(np.float64)
        hl = deconvolve(_as_capture(scale * xl, 1234), xl, FS, 20, 20000)
        il, _ = ir_peak(hl, PRE, PRE + int(0.2 * FS))
        f, H = frequency_response(hl, il, FS)
        band = (f >= 20) & (f <= 19000)
        assert np.max(np.abs(20 * np.log10(np.abs(H[band])) + 6.0206)) <= 0.05
