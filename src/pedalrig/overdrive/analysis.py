"""Overdrive analysis: level transfer from ramps (clipping onset, harmonics, asymmetry), bursts, blend."""

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from ..analysis import PedalAnalysis, band_mean, db, dbu, harmonics, next_pow2, per_cycle


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
        q, l = self.sweep_response(m30_name), self.sweep_response(m10_name)
        f = q["f"]
        norm = lambda r: r["mag_db"] - np.interp(1000, f, r["mag_db"])  # noqa: E731
        thd = l["hd"]["THD"]
        return {"f": f, "quiet_db": norm(q), "loud_db": norm(l),
                "thd_low_40_150_db": band_mean(f, thd, 40, 150),
                "thd_mid_400_2k_db": band_mean(f, thd, 400, 2000)}
