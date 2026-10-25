"""Analysis library for the notebooks: pure numpy/scipy, reads saved sessions, writes nothing.

Level conventions
  dBFS: sine peak amplitude re full scale (a = 1.0 -> 0 dBFS); noise RMS uses the sine
        reference 20*log10(rms*sqrt(2)).
  dBu (send-referred, per manual spec): dBFS + SEND_DBU_AT_0DBFS. Captures are divided by the
        loopback gain G (calibration C5) so they read in the same units as the stimulus.
"""

import numpy as np
from scipy import signal
from scipy.fft import irfft, rfft, rfftfreq

from .config import CLIP, IN_CH, SEND_DBU_AT_0DBFS

# ---------------------------------------------------------------- primitives


def next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def db(x, floor: float = 1e-15):
    return 20 * np.log10(np.maximum(np.abs(x), floor))


def log_grid(f_lo: float, f_hi: float, n: int = 300) -> np.ndarray:
    return np.geomspace(f_lo, f_hi, n)


def harmonics(x: np.ndarray, fs: float, f: float, n: int = 5) -> dict:
    """Least-squares fit of DC + n harmonics of f. A[k-1], phase[k-1] for harmonic k (cosine phase)."""
    t = np.arange(len(x)) / fs
    cols = [np.ones_like(t)]
    for k in range(1, n + 1):
        cols += [np.cos(2 * np.pi * k * f * t), np.sin(2 * np.pi * k * f * t)]
    M = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(M, x, rcond=None)
    a, b = coef[1::2], coef[2::2]
    return {"dc": float(coef[0]), "A": np.hypot(a, b), "phase": np.arctan2(-b, a),
            "residual": x - M @ coef}


def _bandpass(x: np.ndarray, fs: float, lo: float = 20.0, hi: float = 20000.0) -> np.ndarray:
    sos = signal.butter(4, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def thdn_db(x: np.ndarray, fs: float, f: float) -> float:
    """THD+N re fundamental RMS, residual band-limited to 20 Hz-20 kHz."""
    h = harmonics(x, fs, f, n=1)
    res = _bandpass(h["residual"], fs)
    return float(20 * np.log10(np.sqrt(np.mean(res ** 2)) / (h["A"][0] / np.sqrt(2))))


def _band_window(freqs: np.ndarray, f1: float, f2: float, fs: float) -> np.ndarray:
    """Complex band limit for deconvolution. Low edge: causal 4th-order Butterworth high-pass at
    f1/2 (a zero-phase taper there rings for hundreds of ms and the truncated pre-ringing corrupts
    the whole band; this one costs < 0.02 dB at f1). High edge: zero-phase raised cosine over
    [f2, f2*2^(1/3)] (short ringing)."""
    sos = signal.butter(4, f1 / 2, btype="highpass", fs=fs, output="sos")
    _, hp = signal.sosfreqz(sos, worN=freqs, fs=fs)
    up = (freqs <= f2).astype(float)
    hi1 = min(f2 * 2 ** (1 / 3), 0.49 * fs)
    if hi1 > f2:
        m = (freqs > f2) & (freqs <= hi1)
        up[m] = 0.5 + 0.5 * np.cos(np.pi * (freqs[m] - f2) / (hi1 - f2))
    return hp * up


def deconvolve(rec: np.ndarray, stim: np.ndarray, fs: float, f1: float, f2: float) -> np.ndarray:
    """Regularized, band-limited spectral division. Linear IR lands at the recording delay;
    harmonic IRs of a synchronized sweep wrap to the end of the array."""
    N = next_pow2(len(rec) + len(stim))
    X, Y = rfft(stim, N), rfft(rec, N)
    P = np.abs(X) ** 2
    Xinv = np.conj(X) / (P + 1e-8 * P.max()) * _band_window(rfftfreq(N, 1 / fs), f1, f2, fs)
    return irfft(Y * Xinv, N)


def ir_peak(h: np.ndarray, lo: int, hi: int) -> tuple[float, int]:
    """Fractional index of max |h| in [lo, hi) (parabolic interpolation) and its sign."""
    seg = np.abs(h[lo:hi])
    i = int(np.argmax(seg))
    frac = 0.0
    if 0 < i < len(seg) - 1:
        a, b, c = seg[i - 1], seg[i], seg[i + 1]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den != 0 else 0.0
    return lo + i + frac, int(np.sign(h[lo + i]))


def _ir_window(h: np.ndarray, center: float, fs: float, pre_s: float, post_s: float):
    """Circular slice [center-pre, center+post) with half-Hann fade on the last 10%.
    Returns (segment, offset of `center` inside the segment in samples, fractional)."""
    N = len(h)
    pre, post = int(round(pre_s * fs)), int(round(post_s * fs))
    start = int(np.floor(center)) - pre
    seg = h[np.arange(start, start + pre + post) % N].copy()
    nf = max(1, int(0.1 * len(seg)))
    seg[-nf:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(nf) / nf)
    return seg, center - start


def ir_spectrum(h: np.ndarray, center: float, fs: float, pre_s: float, post_s: float,
                nfft: int) -> tuple[np.ndarray, np.ndarray]:
    seg, off = _ir_window(h, center, fs, pre_s, post_s)
    nfft = max(nfft, next_pow2(len(seg)))
    freqs = rfftfreq(nfft, 1 / fs)
    H = rfft(seg, nfft) * np.exp(2j * np.pi * freqs * off / fs)  # remove the pre-offset delay
    return freqs, H


def frequency_response(h: np.ndarray, i1: float, fs: float, nfft: int = 2 ** 17):
    return ir_spectrum(h, i1, fs, 0.005, 0.5, nfft)


def harmonic_irs(h: np.ndarray, i1: float, L: float, fs: float, n_max: int = 5,
                 pre_s: float = 0.002, post_s: float = 0.1) -> list[tuple[np.ndarray, float]]:
    """Windowed IRs for harmonics 1..n_max; harmonic n sits at (i1 - L*ln(n)*fs) mod N."""
    N = len(h)
    return [_ir_window(h, (i1 - L * np.log(n) * fs) % N, fs, pre_s, post_s) for n in range(1, n_max + 1)]


def hd_vs_freq(irs: list[tuple[np.ndarray, float]], fs: float, f_valid_hi: float,
               grid: np.ndarray | None = None, nfft: int = 2 ** 15) -> dict:
    """HD_n(f) = |H_n(n f)| / |H_1(f)| in dB (NaN where n f > f_valid_hi); THD(f) over n = 2..len(irs)."""
    grid = log_grid(20, 20000) if grid is None else grid
    freqs = rfftfreq(nfft, 1 / fs)
    mags = [np.abs(rfft(seg, nfft)) for seg, _ in irs]
    H1 = np.interp(grid, freqs, mags[0])
    HD = {}
    for n in range(2, len(irs) + 1):
        v = db(np.interp(n * grid, freqs, mags[n - 1]) / np.maximum(H1, 1e-15))
        v[n * grid > f_valid_hi] = np.nan
        HD[n] = v
    stack = np.array([10 ** (HD[n] / 10) for n in HD])
    total = np.nansum(stack, axis=0)
    thd = np.full_like(total, np.nan)
    ok = ~np.isnan(HD[2])
    thd[ok] = 10 * np.log10(total[ok])
    return {"f": grid, "HD": HD, "THD": thd}


def smooth_frac_oct(freqs: np.ndarray, H: np.ndarray, frac: int = 12,
                    grid: np.ndarray | None = None) -> np.ndarray:
    """Power average within +-1/(2*frac) octave around each grid frequency; returns dB."""
    grid = log_grid(20, 20000) if grid is None else grid
    cs = np.concatenate([[0.0], np.cumsum(np.abs(H) ** 2)])
    lo = np.searchsorted(freqs, grid * 2 ** (-1 / (2 * frac)))
    hi = np.searchsorted(freqs, grid * 2 ** (1 / (2 * frac)), side="right")
    near = np.clip(np.searchsorted(freqs, grid), 0, len(freqs) - 1)
    empty = hi <= lo
    lo, hi = np.where(empty, near, lo), np.where(empty, near + 1, hi)
    return 10 * np.log10((cs[hi] - cs[lo]) / (hi - lo) + 1e-30)


def interp_phase_deg(freqs: np.ndarray, H: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ph = np.interp(grid, freqs, np.unwrap(np.angle(H)))
    return np.degrees(ph - 2 * np.pi * np.round(ph[np.argmin(np.abs(grid - 1000))] / (2 * np.pi)))


def noise_stats(x: np.ndarray, fs: float) -> dict:
    xb = _bandpass(x - np.mean(x), fs)
    f, P = signal.welch(x, fs, window="hann", nperseg=int(fs), scaling="spectrum")
    hum = {}
    for hz in (50, 60, 100, 120, 150, 180):
        i = int(np.argmin(np.abs(f - hz)))
        hum[hz] = float(10 * np.log10(2 * P[max(i - 1, 0):i + 2].max() + 1e-30))
    return {"rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(xb ** 2)) * np.sqrt(2) + 1e-30)),
            "dc": float(np.mean(x)), "psd_f": f, "psd_dbfs": 10 * np.log10(2 * P + 1e-30), "hum": hum}


def per_cycle(x: np.ndarray, fs: float, f: float, cycles: int = 4, hop_cycles: int = 1,
              n_harm: int = 9) -> dict:
    """Windowed DFT over integer periods. A[:, k-1] = peak amplitude of harmonic k."""
    p = fs / f
    if abs(p - round(p)) > 1e-9:
        raise ValueError(f"{f} Hz is not an integer period at {fs} Hz")
    p = int(round(p))
    w, hop = cycles * p, hop_cycles * p
    nwin = (len(x) - w) // hop + 1
    idx = np.arange(nwin)[:, None] * hop + np.arange(w)[None, :]
    segs = x[idx]
    X = np.fft.rfft(segs, axis=1)
    bins = cycles * np.arange(1, n_harm + 1)
    bins = bins[bins < X.shape[1]]
    return {"t": (np.arange(nwin) * hop + w / 2) / fs, "A": 2 * np.abs(X[:, bins]) / w,
            "phase": np.angle(X[:, bins]), "peak_pos": segs.max(axis=1), "peak_neg": segs.min(axis=1),
            "mean": segs.mean(axis=1), "rms": np.sqrt(np.mean(segs ** 2, axis=1))}


def segment(rec_col: np.ndarray, latency: int, pre_s: float, fs: float, length: int) -> np.ndarray:
    s = int(round(pre_s * fs)) + int(latency)
    return rec_col[s:s + length]


def correlation_lag(rec: np.ndarray, stim: np.ndarray) -> int:
    """Index in `rec` where `stim` starts (max |cross-correlation|)."""
    c = signal.fftconvolve(rec, stim[::-1], mode="full")
    return int(np.argmax(np.abs(c))) - (len(stim) - 1)


def band_mean(f: np.ndarray, v: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f <= hi) & np.isfinite(v)
    return float(np.mean(v[m])) if m.any() else float("nan")


def dbu(dbfs):
    return np.asarray(dbfs) + SEND_DBU_AT_0DBFS


# ---------------------------------------------------------------- session helpers


def in_col(sess, name: str) -> np.ndarray:
    """Recorded USB In 8 (INPUT R) column of capture `name`."""
    info = sess.capture_info(name)
    return sess.load_capture(name)[:, info["in_channels"].index(IN_CH)]


def _sweep_ir(sess, name: str, x: np.ndarray | None = None):
    info = sess.capture_info(name)
    p = sess.stim_params(info["stimulus"])
    x = in_col(sess, name) if x is None else x
    h = deconvolve(x, sess.load_stimulus(info["stimulus"]), sess.meta["fs"], p["f1"], p["f2"])
    return h, p


def _i1(sess, h: np.ndarray) -> tuple[float, int]:
    """Linear-IR peak of a sweep capture: searched over the whole post-roll, since the
    stream latency (~270 ms measured at 48 kHz) is far larger than the analog delay."""
    fs = sess.meta["fs"]
    pre = int(round(sess.meta["pre_s"] * fs))
    return ir_peak(h, pre, pre + int(sess.meta["post_s"] * fs))


# ---------------------------------------------------------------- calibration


def calibration_summary(sess) -> dict:
    """Everything the pedal analysis needs from a cable-loopback session (computed in memory)."""
    fs, pre_s = sess.meta["fs"], sess.meta["pre_s"]
    pre = int(round(pre_s * fs))
    names = set(sess.capture_names())
    out: dict = {"session": str(sess.path), "checks": {}, "curves": {}}
    chk = out["checks"]

    # C4 latency & polarity
    lat_names = sorted(n for n in names if n.startswith("latency_"))
    if lat_names:
        runs, pols = [], []
        for n in lat_names:
            h, _ = _sweep_ir(sess, n)
            i1, pol = _i1(sess, h)
            runs.append(i1 - pre)
            pols.append(pol)
        out["latency_runs"] = [float(r) for r in runs]
        out["latency_samples"] = int(round(float(np.median(runs))))
        out["latency_ms"] = out["latency_samples"] / fs * 1e3
        out["latency_stable"] = bool(max(runs) - min(runs) <= 1.0)
        out["polarity"] = 1 if sum(pols) >= 0 else -1
        chk["latency"] = "PASS" if out["latency_stable"] else "WARN"
    lat = out.get("latency_samples", 0)

    # C2 routing
    if "routing_1k" in names:
        rec = sess.load_capture("routing_1k")
        seg = slice(pre + lat + int(0.5 * fs), pre + lat + int(1.5 * fs))
        lv = [float(db(harmonics(rec[seg, c], fs, 1000, n=1)["A"][0])) for c in range(rec.shape[1])]
        chans = sess.capture_info("routing_1k")["in_channels"]
        out["routing_dbfs"] = dict(zip(chans, lv))
        r = out["routing_dbfs"]
        ok = r[8] >= -80 and r[8] == max(r[c] for c in (5, 6, 7, 8)) and r[7] <= r[8] - 40
        chk["routing"] = "PASS" if ok else "FAIL"

    # C3 noise
    if "noise" in names:
        ns = noise_stats(in_col(sess, "noise")[pre:pre + int(5 * fs)], fs)
        out["noise_dbfs"], out["dc"], out["hum"] = ns["rms_dbfs"], ns["dc"], ns["hum"]
        out["curves"]["noise_psd"] = (ns["psd_f"], ns["psd_dbfs"])
        bad = ns["rms_dbfs"] > -80 or max(ns["hum"].values()) > -90
        chk["noise"] = "WARN" if bad else "PASS"

    # C5 levels
    if "steps_1k" in names:
        p = sess.stim_params("steps_1k")
        x = in_col(sess, "steps_1k")
        n_step, n_seg = int(round(p["step_s"] * fs)), int(round(0.6 * fs))
        lat_s = correlation_lag(x, sess.load_stimulus("steps_1k")) - pre
        rows = []
        for lv, st in zip(p["levels_dbfs"], p["step_starts"]):
            s = pre + lat_s + st + (n_step - n_seg) // 2
            seg = x[s:s + n_seg]
            hm = harmonics(seg, fs, 1000, n=5)
            out_db = float(db(hm["A"][0]))
            rows.append({"in_dbfs": lv, "in_dbu": lv + SEND_DBU_AT_0DBFS, "out_dbfs": out_db,
                         "gain_db": out_db - lv, "thdn_db": thdn_db(seg, fs, 1000),
                         **{f"hd{k}_db": float(db(hm["A"][k - 1] / hm["A"][0])) for k in range(2, 6)},
                         "clipped": bool(np.any(np.abs(x[pre + lat_s + st:pre + lat_s + st + n_step]) >= CLIP))})
        out["levels"] = rows
        mid = [r for r in rows if -40 <= r["in_dbfs"] <= -6]
        out["G_db"] = float(np.median([r["gain_db"] for r in mid]))
        out["G_lin"] = float(10 ** (out["G_db"] / 20))
        clean = [r["in_dbfs"] for r in rows if not r["clipped"] and r["thdn_db"] < -60]
        out["max_clean_dbfs"] = max(clean) if clean else None
        r20 = next(r for r in rows if r["in_dbfs"] == -20)
        ok = all(abs(r["gain_db"] - out["G_db"]) <= 0.1 for r in mid) and r20["thdn_db"] <= -60
        chk["levels"] = "PASS" if ok else "WARN"

    # C6 wideband
    if "wide_sweep" in names:
        h, p = _sweep_ir(sess, "wide_sweep")
        i1, _ = _i1(sess, h)
        f, H = frequency_response(h, i1, fs)
        grid = log_grid(5, 0.49 * fs, 500)
        mag = smooth_frac_oct(f, H, 12, grid)
        mag -= np.interp(1000, grid, mag)
        out["curves"]["wide_fr"] = (grid, mag, interp_phase_deg(f, H, grid))
        i1k = int(np.argmin(np.abs(grid - 1000)))

        def corner(thr, up):
            rng = range(i1k, len(grid)) if up else range(i1k, -1, -1)
            for i in rng:
                if mag[i] < thr:
                    return float(grid[i])
            return None

        out["bw_lo_3db"], out["bw_hi_3db"], out["bw_hi_1db"] = corner(-3, False), corner(-3, True), corner(-1, True)
        band = (grid >= 20) & (grid <= 20000)
        out["ripple_20_20k_db"] = float(np.max(np.abs(mag[band])))
        out["ripple_minmax_db"] = (float(mag[band].min()), float(mag[band].max()))
        out["curves"]["wide_hd"] = hd_vs_freq(harmonic_irs(h, i1, p["L"], fs), fs, p["f2"],
                                              log_grid(20, 20000))
        chk["wideband"] = "PASS" if out["ripple_20_20k_db"] <= 1.0 else "WARN"

    # C8 repeatability vs C7 ref_sweep_m20
    reps = sorted(n for n in names if n.startswith("repeat_sweep_m20_"))
    if reps and "ref_sweep_m20" in names:
        grid = log_grid(20, 20000)
        curves, peaks = {}, {}
        for n in ["ref_sweep_m20", *reps]:
            h, _ = _sweep_ir(sess, n)
            i1, _ = _i1(sess, h)
            f, H = frequency_response(h, i1, fs)
            curves[n], peaks[n] = smooth_frac_oct(f, H, 12, grid), i1
        out["repeat_dev_db"] = float(max(np.max(np.abs(curves[n] - curves["ref_sweep_m20"])) for n in reps))
        out["repeat_latency_equal"] = bool(max(peaks.values()) - min(peaks.values()) <= 1.0)
        out["curves"]["repeat"] = (grid, {n: c - curves["ref_sweep_m20"] for n, c in curves.items()})
        ok = out["repeat_dev_db"] <= 0.1 and out["repeat_latency_equal"]
        chk["repeat"] = "PASS" if ok else "WARN"

    return out


# ---------------------------------------------------------------- pedal


class PedalAnalysis:
    """Pedal captures referenced to the calibration loopback: x polarity / G_lin, divided by the
    same-stimulus reference where a transfer function is computed."""

    def __init__(self, pedal_sess, cal_summary: dict, cal_sess):
        self.ps, self.cal, self.cs = pedal_sess, cal_summary, cal_sess
        self.fs = pedal_sess.meta["fs"]
        self.pre = int(round(pedal_sess.meta["pre_s"] * self.fs))
        for k in ("G_lin", "polarity", "latency_samples"):
            if k not in cal_summary:
                raise ValueError(f"calibration summary lacks '{k}': run uv run pedalrig calibrate")
        need = sorted({c["stimulus"] for c in pedal_sess.meta["captures"]})
        missing = [f"ref_{s}" for s in need if f"ref_{s}" not in cal_sess.capture_names()]
        if missing:
            raise ValueError(f"missing loopback refs {missing}: run uv run pedalrig calibrate")
        self.scale = cal_summary["polarity"] / cal_summary["G_lin"]
        self.latency_stable = cal_summary.get("latency_stable")
        self.f_valid_hi = min(20000.0, cal_summary.get("bw_hi_3db") or 20000.0)

    def _y(self, sess, name):
        return in_col(sess, name) * self.scale

    def _ref(self, name):
        return f"ref_{self.ps.capture_info(name)['stimulus']}"

    def _latency(self, y, stim) -> int:
        """Per-capture coarse alignment. Stream latency can change between stream opens, so the
        calibration value is not reused. For sines an error of whole periods is harmless."""
        return correlation_lag(y, stim) - self.pre

    def knobs(self, name):
        return self.ps.capture_info(name)["knobs"]

    # ---- sweeps (T2, T3, T5)
    def sweep_response(self, name: str) -> dict:
        fs = self.fs
        h_ref, p = _sweep_ir(self.cs, self._ref(name), self._y(self.cs, self._ref(name)))
        h_cap, _ = _sweep_ir(self.ps, name, self._y(self.ps, name))
        i1_ref, _ = _i1(self.cs, h_ref)
        i1_cap, _ = _i1(self.ps, h_cap)
        f, Hc = frequency_response(h_cap, i1_cap, fs)
        _, Hr = frequency_response(h_ref, i1_ref, fs)
        H = Hc / Hr
        # stop 1/12 octave short of f2: the smoothing window would reach into the taper/fade-out,
        # where the cap/ref ratio is noise (measured: up to 1.3 dB at 20 kHz vs <= 0.02 dB below)
        grid = log_grid(20, min(20000.0, p["f2"] * 2 ** (-1 / 12)))
        fv = min(self.f_valid_hi, p["f2"])
        return {"name": name, "knobs": self.knobs(name), "level_dbfs": p["level_dbfs"], "f": grid,
                "mag_db": smooth_frac_oct(f, H, 12, grid), "phase_deg": interp_phase_deg(f, H, grid),
                "hd": hd_vs_freq(harmonic_irs(h_cap, i1_cap, p["L"], fs), fs, fv, grid),
                "ref_hd": hd_vs_freq(harmonic_irs(h_ref, i1_ref, p["L"], fs), fs, fv, grid)}

    # ---- ramps (T1)
    def ramp_transfer(self, name: str) -> dict:
        fs = self.fs
        info = self.ps.capture_info(name)
        p = self.ps.stim_params(info["stimulus"])
        f = p["f"]
        x = self.ps.load_stimulus(info["stimulus"])
        yr = self._y(self.ps, name)
        lat = self._latency(yr, x)
        y = segment(yr, lat, self.ps.meta["pre_s"], fs, len(x))

        px, py = per_cycle(x, fs, f), per_cycle(y, fs, f)
        # drop windows overlapping the 10 ms fades (by time: at 3 kHz that is ~30 windows)
        keep = (px["t"] > 0.015) & (px["t"] < len(x) / fs - 0.015)
        in_db, out_db = db(px["A"][keep, 0]), db(py["A"][keep, 0])
        n20 = max(3, int(0.2 * len(in_db)))
        g = float(np.median(out_db[:n20] - in_db[:n20]))
        short = (in_db + g) - out_db
        onset = None
        for i in range(len(short) - 5):
            if np.all(short[i:i + 5] >= 1.0):
                onset = float(dbu(in_db[i]))
                break
        A = py["A"][keep]
        kmax = min(A.shape[1], int(self.f_valid_hi // f))
        rel = A[:, 1:kmax] / A[:, [0]]
        thd = 10 * np.log10(np.sum(rel ** 2, axis=1) + 1e-30)

        # instantaneous transfer: undo linear phase / inversion measured at low level
        n1 = int(fs)
        phi = harmonics(y[:n1], fs, f, 1)["phase"][0] - harmonics(x[:n1], fs, f, 1)["phase"][0]
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        inverted = bool(np.cos(phi) < 0)
        ys = -y if inverted else y.copy()
        if inverted:
            phi = (phi - np.pi + np.pi) % (2 * np.pi) - np.pi
        tau = phi / (2 * np.pi * f)
        N = next_pow2(len(ys) * 2)
        Y = rfft(ys, N) * np.exp(-2j * np.pi * rfftfreq(N, 1 / fs) * tau)
        ys = irfft(Y, N)[:len(ys)]

        return {"name": name, "knobs": self.knobs(name), "f": f,
                "in_dbu": dbu(in_db), "out_dbu": dbu(out_db), "linear_gain_db": g,
                "onset_1db_dbu": onset,
                "hd2_db": db(rel[:, 0]) if rel.shape[1] >= 1 else None,
                "hd3_db": db(rel[:, 1]) if rel.shape[1] >= 2 else None, "thd_db": thd,
                "asym_db": db(py["peak_pos"][keep]) - db(py["peak_neg"][keep]),
                "x": x, "y": ys, "phase_rad": float(phi), "inverted": inverted}

    # ---- bursts (T4)
    def _bursts(self, sess, name):
        fs = self.fs
        info = sess.capture_info(name)
        p = sess.stim_params(info["stimulus"])
        x = sess.load_stimulus(info["stimulus"])
        yr = self._y(sess, name)
        lat = self._latency(yr, x)
        y = segment(yr, lat, sess.meta["pre_s"], fs, len(x))
        n_on = int(round(p["on_s"] * fs))
        env, mean, asym = [], [], []
        for st in p["burst_starts"]:
            pc = per_cycle(y[st:st + n_on], fs, p["f"], cycles=1, hop_cycles=1, n_harm=1)
            env.append(dbu(db(pc["rms"] * np.sqrt(2))))
            mean.append(pc["mean"] / np.maximum(pc["peak_pos"], 1e-12))
            asym.append(db(pc["peak_pos"]) - db(pc["peak_neg"]))
        env = np.array(env)
        # second cycle vs last full cycle (index -2: the final cycle carries the 1 ms fade-out)
        return {"t_ms": pc["t"] * 1e3, "env_dbu": env, "mean_rel": np.array(mean),
                "asym_db": np.array(asym), "sag_db": env[:, -2] - env[:, 1]}

    def burst_dynamics(self, name: str) -> dict:
        return {"name": name, "knobs": self.knobs(name), "cap": self._bursts(self.ps, name),
                "ref": self._bursts(self.cs, self._ref(name))}

    # ---- composite metrics
    def volume_check(self, names: list[str]) -> dict:
        rs = {n: self.sweep_response(n) for n in names}
        noon = next(n for n in names if n.endswith("noon"))
        f = rs[noon]["f"]
        band = (f >= 50) & (f <= 10000)
        out = {}
        for n, r in rs.items():
            d = r["mag_db"] - rs[noon]["mag_db"]
            d1k = float(np.interp(1000, f, d))
            out[n] = {"level_re_noon_db": d1k, "shape_dev_db": float(np.max(np.abs(d[band] - d1k)))}
        return {"responses": rs, "metrics": out}

    def blend_metric(self, m30_name: str, m10_name: str) -> dict:
        q, l = self.sweep_response(m30_name), self.sweep_response(m10_name)
        f = q["f"]
        norm = lambda r: r["mag_db"] - np.interp(1000, f, r["mag_db"])  # noqa: E731
        thd = l["hd"]["THD"]
        return {"f": f, "quiet_db": norm(q), "loud_db": norm(l),
                "thd_low_40_150_db": band_mean(f, thd, 40, 150),
                "thd_mid_400_2k_db": band_mean(f, thd, 400, 2000)}
