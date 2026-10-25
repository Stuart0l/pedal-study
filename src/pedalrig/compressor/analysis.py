"""Compressor analysis: static compression curve, attack/release timing, decay (sustain), noise,
compression vs frequency. Levels in dBu are send-referred (see pedalrig.analysis)."""

import numpy as np

from ..analysis import PedalAnalysis, db, dbu, harmonics, noise_stats, per_cycle
from ..config import SEND_DBU_AT_0DBFS

GR_THRESHOLD_DB = 1.0   # threshold := input level where gain reduction reaches 1 dB
MIN_CHANGE_DB = 1.0     # level jumps that move the gain less than this are not timed
RUNOUT_SLOPE = 0.8      # past the knee, a local slope above this means the gain reduction ran out


def fit_static(in_dbu: np.ndarray, out_dbu: np.ndarray) -> dict:
    """Compression-curve figures from settled (input, output) levels, sorted by input.

    low-level gain: median gain of the 3 quietest points (their slope says if they are uncompressed)
    threshold: input where gain reduction (re low-level gain) first reaches 1 dB and stays there
    ratio: 1 / slope of output vs input over the (up to 4) loudest points at least 6 dB above the
           threshold, i.e. past the knee, and below the run-out (inf when the output is flat; None
           with < 2 such points)
    run-out: input where the curve turns back toward 1:1 (a 3 dB step past the knee raises the
           output by > 0.8 × 3 dB): the gain reduction has hit its limit (`gr_limit_db`); louder
           input passes uncompressed
    """
    in_dbu, out_dbu = np.asarray(in_dbu, float), np.asarray(out_dbu, float)
    gain = out_dbu - in_dbu
    g0 = float(np.median(gain[:3]))
    low_slope = float(np.polyfit(in_dbu[:3], out_dbu[:3], 1)[0])
    gr = g0 - gain
    thr = None
    above = gr >= GR_THRESHOLD_DB
    if above.any():
        below = np.nonzero(~above)[0]
        i = 0 if len(below) == 0 else below[-1] + 1
        if i < len(gr):
            if i == 0:
                thr = float(in_dbu[0])
            else:  # interpolate the 1 dB crossing between the last point below and the first above
                a, b = gr[i - 1], gr[i]
                thr = float(in_dbu[i - 1] + (GR_THRESHOLD_DB - a) / (b - a) * (in_dbu[i] - in_dbu[i - 1]))
    slope = np.diff(out_dbu) / np.diff(in_dbu)   # slope[j]: between points j and j+1
    use = np.nonzero(in_dbu >= thr + 6)[0] if thr is not None else np.array([], int)
    runout = None
    for j in use[:-1]:
        if slope[j] > RUNOUT_SLOPE:
            runout = float(in_dbu[j])
            use = use[use < j]  # point j may already sit on the limit
            break
    use = use[-4:]
    top_slope = float(np.polyfit(in_dbu[use], out_dbu[use], 1)[0]) if len(use) >= 2 else None
    ratio = None if top_slope is None else (1.0 / top_slope if top_slope > 0.02 else float("inf"))
    return {"low_gain_db": g0, "low_slope": low_slope, "low_linear": low_slope >= 0.95,
            "gr_db": gr, "threshold_dbu": thr, "top_slope": top_slope, "ratio": ratio,
            "runout_dbu": runout, "gr_limit_db": float(gr.max()) if runout is not None else None,
            "max_gr_db": float(gr.max()), "local_slope": slope,
            "out_span_top30_db": float(out_dbu[-1] - np.interp(in_dbu[-1] - 30, in_dbu, out_dbu))}


def settle_times(t: np.ndarray, g: np.ndarray, g_start: float, g_end: float,
                 fracs=(0.63, 0.9)) -> dict:
    """Time after a level jump until the gain (dB) has covered `frac` of its change and stays
    there, for each frac. None when the change is below MIN_CHANGE_DB (nothing to time) or the
    gain never settles within `t`. `t` is time since the jump, `g` the gain trace from the jump on."""
    d = g_end - g_start
    if abs(d) < MIN_CHANGE_DB:
        return {fr: None for fr in fracs}
    prog = (np.asarray(g) - g_start) / d
    out = {}
    for fr in fracs:
        short = np.nonzero(prog < fr)[0]
        if len(short) == 0:
            out[fr] = float(t[0])
        elif short[-1] + 1 < len(t):
            out[fr] = float(t[short[-1] + 1])
        else:
            out[fr] = None
    return out


class CompressorAnalysis(PedalAnalysis):
    # ---- C1 / C6: held-level staircase
    def static_curve(self, name: str) -> dict:
        """Settled output per staircase step: the last 0.5 s of each 2 s step (before its 10 ms
        fade). `settle_db` compares it with the 0.5 s before; |settle_db| > 0.3 means the pedal
        was still moving (release/attack slower than the hold)."""
        fs = self.fs
        x, y, p = self.aligned(name)
        f = p["f"]
        n_step, n_win, n_fade = (int(round(v * fs)) for v in (p["step_s"], 0.5, 0.01))
        kmax = max(1, min(9, int(self.f_valid_hi // f)))
        rows = []
        for lv, st in zip(p["levels_dbfs"], p["step_starts"]):
            b = st + n_step - n_fade
            a = b - n_win
            a_in = harmonics(x[a:b], fs, f, 1)["A"][0]
            A = harmonics(y[a:b], fs, f, kmax)["A"]
            prev = harmonics(y[a - n_win:a], fs, f, 1)["A"][0]
            rows.append({"level_dbfs": lv, "in_dbu": float(dbu(db(a_in))), "out_dbu": float(dbu(db(A[0]))),
                         "thd_db": float(10 * np.log10(np.sum((A[1:] / A[0]) ** 2) + 1e-30)) if kmax > 1 else None,
                         "settle_db": float(db(A[0]) - db(prev))})
        in_dbu = np.array([r["in_dbu"] for r in rows])
        out_dbu = np.array([r["out_dbu"] for r in rows])
        return {"name": name, "knobs": self.knobs(name), "f": f, "rows": rows, "in_dbu": in_dbu,
                "out_dbu": out_dbu, "thd_db": np.array([np.nan if r["thd_db"] is None else r["thd_db"] for r in rows]),
                "max_settle_db": float(max(abs(r["settle_db"]) for r in rows)), **fit_static(in_dbu, out_dbu)}

    # ---- C2 / C3: level jumps
    def level_steps_response(self, name: str) -> dict:
        """Per-cycle gain (output − input, dB) around every level jump. Attack = jump up, release
        = jump down. Times are to 63 % / 90 % of the gain change (time resolution: one period).
        g_start = gain over the last 20 ms before the jump, g_end = over the last 0.1 s of the
        segment after it; `settle_db` = g_end vs the 0.1 s before that (non-zero: not settled).
        Attack only: `overshoot_db` = highest output in the first 50 ms above the settled level;
        `dip_db` / `dip_s` = how far (and when) the gain swings below its settled value after the
        attack has reached 90 % (over-compression before it recovers)."""
        fs = self.fs
        x, y, p = self.aligned(name)
        f = p["f"]
        px = per_cycle(x, fs, f, cycles=1, hop_cycles=1, n_harm=1)
        py = per_cycle(y, fs, f, cycles=1, hop_cycles=1, n_harm=1)
        t = px["t"]
        in_db, out_db = db(px["A"][:, 0]), db(py["A"][:, 0])
        gain = out_db - in_db
        segs = p["segments"]
        fade = 0.012

        def med(lo, hi):
            m = (t >= lo) & (t < hi)
            return float(np.median(gain[m]))

        trans = []
        for prev, seg in zip(segs[:-1], segs[1:]):
            ts, te = seg["start"] / fs, (seg["start"] + seg["n"]) / fs
            if seg is segs[-1]:
                te -= fade
            g_start, g_end = med(ts - 0.02, ts), med(te - 0.1, te)
            after = (t >= ts) & (t < te)
            tt, gg = t[after] - ts, gain[after]
            up = seg["level_dbfs"] > prev["level_dbfs"]
            st = settle_times(tt, gg, g_start, g_end)
            view = (t >= ts - 0.05) & (t < te)
            out_end = float(np.median(out_db[(t >= te - 0.1) & (t < te)]))
            first = (t >= ts) & (t < ts + 0.05)
            late = tt >= st[0.9] if st[0.9] is not None else np.zeros(len(tt), bool)
            trans.append({"kind": "attack" if up else "release", "t_step": ts,
                          "hold_s": (prev["n"] / fs), "level_from": prev["level_dbfs"], "level_to": seg["level_dbfs"],
                          "g_start_db": g_start, "g_end_db": g_end, "gain_change_db": g_end - g_start,
                          "t63_s": st[0.63], "t90_s": st[0.9],
                          "overshoot_db": float(out_db[first].max() - out_end) if up else None,
                          "dip_db": float(g_end - gg[late].min()) if up and late.any() else None,
                          "dip_s": float(tt[late][np.argmin(gg[late])]) if up and late.any() else None,
                          "settle_db": g_end - med(te - 0.2, te - 0.1),
                          "t_rel": t[view] - ts, "gain_db": gain[view], "out_dbu": dbu(out_db[view])})
        return {"name": name, "knobs": self.knobs(name), "f": f, "t": t, "in_dbu": dbu(in_db),
                "out_dbu": dbu(out_db), "gain_db": gain, "transitions": trans}

    @staticmethod
    def timing_summary(res: dict) -> dict:
        """Median and spread of attack and release times over the repeated jumps."""
        out = {}
        for kind in ("attack", "release"):
            tr = [d for d in res["transitions"] if d["kind"] == kind]
            for k in ("t63_s", "t90_s", "gain_change_db", "overshoot_db", "dip_db", "dip_s"):
                v = [d[k] for d in tr if d[k] is not None]
                out[f"{kind}_{k}"] = float(np.median(v)) if v else None
                out[f"{kind}_{k}_spread"] = float(np.ptp(v)) if len(v) > 1 else None
        return out

    # ---- C9: plucked-note decay
    def decay_response(self, name: str) -> dict:
        """Per-cycle input/output envelope of a decaying note. `out_time_20db_s` = time after the
        note's start (re its level at 0.1 s) until the output has fallen 20 dB for good; the
        input takes 2 s at 10 dB/s. Decay rates are fitted over 0.5-3 s."""
        fs = self.fs
        x, y, p = self.aligned(name)
        f = p["f"]
        px = per_cycle(x, fs, f, cycles=1, hop_cycles=1, n_harm=1)
        py = per_cycle(y, fs, f, cycles=1, hop_cycles=1, n_harm=1)
        t = px["t"]
        keep = (t > 0.003) & (t < p["dur"] - 0.012)
        t, in_dbu, out_dbu = t[keep], dbu(db(px["A"][keep, 0])), dbu(db(py["A"][keep, 0]))

        def at_01(v):
            return float(np.median(v[(t >= 0.08) & (t < 0.12)]))

        def time_to_drop(v, drop=20.0):
            still = np.nonzero(v >= at_01(v) - drop)[0]
            return float(t[still[-1] + 1]) if still[-1] + 1 < len(t) else None

        fit = (t >= 0.5) & (t <= 3.0)
        return {"name": name, "knobs": self.knobs(name), "t": t, "in_dbu": in_dbu, "out_dbu": out_dbu,
                "in_time_20db_s": time_to_drop(in_dbu), "out_time_20db_s": time_to_drop(out_dbu),
                "in_rate_db_s": float(np.polyfit(t[fit], in_dbu[fit], 1)[0]),
                "out_rate_db_s": float(np.polyfit(t[fit], out_dbu[fit], 1)[0]),
                "attack_peak_db": float(out_dbu[t < 0.05].max() - at_01(out_dbu))}

    # ---- C7: noise
    def noise(self, name: str) -> dict:
        """Output noise with no input (4.5 s from the silent stimulus) vs the loopback floor."""
        fs = self.fs
        sl = slice(self.pre + int(0.25 * fs), self.pre + int(4.75 * fs))
        cap = noise_stats(self._y(self.ps, name)[sl], fs)
        ref = noise_stats(self._y(self.cs, self._ref(name))[sl], fs)
        return {"name": name, "knobs": self.knobs(name),
                "noise_dbu": cap["rms_dbfs"] + SEND_DBU_AT_0DBFS, "floor_dbu": ref["rms_dbfs"] + SEND_DBU_AT_0DBFS,
                "hum": cap["hum"], "psd_f": cap["psd_f"], "psd_dbfs": cap["psd_dbfs"], "floor_psd_dbfs": ref["psd_dbfs"]}

    # ---- C4: compression vs frequency
    def compression_vs_freq(self, small_name: str, loud_name: str) -> dict:
        """Gain reduction at each frequency: small-signal sweep gain − loud sweep gain. Flat =
        the detector treats all frequencies alike; larger at some frequencies = it reacts more to them."""
        s, l = self.sweep_response(small_name), self.sweep_response(loud_name)
        return {"f": s["f"], "gr_db": s["mag_db"] - l["mag_db"], "small": s, "loud": l}
