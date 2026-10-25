"""Overdrive analysis: level transfer from ramps (clipping onset, harmonics, asymmetry), bursts, blend."""

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from ..analysis import PedalAnalysis, band_mean, db, dbu, harmonics, next_pow2, per_cycle

# parallel clean path (Q7): past the knee the output grows again at >= 0.5 dB/dB and THD falls >= 3 dB
BLEND_TOP_SLOPE = 0.5
BLEND_THD_DROP_DB = 3.0


class OverdriveAnalysis(PedalAnalysis):
    # ---- ramps (T1)
    def ramp_transfer(self, name: str) -> dict:
        fs = self.fs
        x, y, p = self.aligned(name)  # x: what the pedal actually received
        f = p["f"]

        px, py = per_cycle(x, fs, f), per_cycle(y, fs, f)
        # drop windows overlapping the 10 ms fades (by time: at 3 kHz that is ~30 windows)
        keep = (px["t"] > 0.015) & (px["t"] < len(x) / fs - 0.015)
        in_db, out_db = db(px["A"][keep, 0]), db(py["A"][keep, 0])
        # Clean (linear) gain from the ramp's own quiet start: its first 5 dB (from -45.5 dBu). The
        # slope of output vs input dB there verifies it is linear (1.0); below 0.95 the pedal is
        # already compressing at the start, so the clean gain is a lower bound and the onset an
        # upper bound (measured: 3 kHz at Gain noon has slope 0.80 and -31 dB THD at -45 dBu).
        start = in_db <= in_db[0] + 5.0
        g = float(np.median(out_db[start] - in_db[start]))
        start_slope = float(np.polyfit(in_db[start], out_db[start], 1)[0])
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
                "onset_1db_dbu": onset, "start_slope": start_slope, "start_linear": start_slope >= 0.95,
                "hd2_db": db(rel[:, 0]) if rel.shape[1] >= 1 else None,
                "hd3_db": db(rel[:, 1]) if rel.shape[1] >= 2 else None, "thd_db": thd,
                "asym_db": db(py["peak_pos"][keep]) - db(py["peak_neg"][keep]),
                "x": x, "y": ys, "phase_rad": float(phi), "inverted": inverted}

    # ---- bursts (T4)
    def _bursts(self, sess, name):
        fs = self.fs
        _, y, p = self.aligned(name, sess)
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

    def blend_metric(self, m30_name: str, m10_name: str) -> dict:
        """Clean bass blend (Q6): responses normalized at 1 kHz and loud-level THD in the bass vs
        the mids. Designs that keep the low end out of the clipper show much lower bass THD."""
        q, l = self.sweep_response(m30_name), self.sweep_response(m10_name)
        f = q["f"]
        norm = lambda r: r["mag_db"] - np.interp(1000, f, r["mag_db"])  # noqa: E731
        thd = l["hd"]["THD"]
        return {"f": f, "quiet_db": norm(q), "loud_db": norm(l),
                "thd_low_40_150_db": band_mean(f, thd, 40, 150),
                "thd_mid_400_2k_db": band_mean(f, thd, 400, 2000)}

    @staticmethod
    def clean_blend(r: dict) -> dict | None:
        """Signs of an unclipped path mixed with the clipped one, from a ramp_transfer result
        (None if the ramp never reached its 1 dB onset).

        A plain clipper stays at its ceiling past the knee: output slope ~0, THD stays high. With
        a clean path summed in, the clipped part saturates while the clean part keeps growing, so
        the slope recovers toward 1 and THD falls again at high input. Slopes are over 3 dB of
        input, THD is the median over 1 dB. `clean_gain_max_db` = output − input at the top: an
        upper bound for the clean path's gain (the clipped part still adds to the output)."""
        onset = r["onset_1db_dbu"]
        if onset is None:
            return None
        i, o, thd = r["in_dbu"], r["out_dbu"], r["thd_db"]
        top = float(i[-1])
        xs = np.arange(onset, top - 3.0 + 1e-9, 0.5)
        if len(xs) < 2:
            return None
        slope = np.array([(np.interp(x + 3, i, o) - np.interp(x, i, o)) / 3 for x in xs])
        thd_med = np.array([np.median(thd[(i >= x) & (i < x + 1)]) for x in np.arange(onset, top - 1 + 1e-9, 0.5)])
        top_slope = float(slope[-1])
        thd_peak, thd_top = float(thd_med.max()), float(thd_med[-1])
        return {"name": r["name"], "min_slope": float(slope.min()), "top_slope": top_slope,
                "thd_peak_db": thd_peak, "thd_top_db": thd_top, "thd_drop_db": thd_peak - thd_top,
                "clean_gain_max_db": float(o[-1] - i[-1]),
                "clean_path": bool(top_slope >= BLEND_TOP_SLOPE and thd_peak - thd_top >= BLEND_THD_DROP_DB)}
