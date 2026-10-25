"""Test-signal generators. All levels are sine peak amplitudes in dBFS."""

from dataclasses import dataclass, field

import numpy as np

from .config import FS


@dataclass
class Stimulus:
    name: str
    data: np.ndarray  # float32, 1-D, unpadded
    params: dict = field(default_factory=dict)


def _amp(level_dbfs: float) -> float:
    return 10.0 ** (level_dbfs / 20.0)


def _fade(n: int) -> np.ndarray:
    """Raised-cosine rise of n samples (0 -> 1)."""
    if n <= 0:
        return np.ones(0)
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)


def _apply_fades(x: np.ndarray, fade_in_s: float, fade_out_s: float) -> np.ndarray:
    ni, no = int(round(fade_in_s * FS)), int(round(fade_out_s * FS))
    x[:ni] *= _fade(ni)
    if no:
        x[-no:] *= _fade(no)[::-1]
    return x


def sync_sweep(name: str, f1: float, f2: float, T: float, level_dbfs: float) -> Stimulus:
    """Synchronized exponential sweep (Novak et al.): f1*L is an integer, so harmonic
    impulse responses separate cleanly at times -L*ln(n)."""
    L = round(T * f1 / np.log(f2 / f1)) / f1
    T_act = L * np.log(f2 / f1)
    t = np.arange(int(np.floor(T_act * FS))) / FS
    x = _amp(level_dbfs) * np.sin(2 * np.pi * f1 * L * np.exp(t / L))
    x = _apply_fades(x, 0.1, 0.005)
    params = {"kind": "sync_sweep", "f1": float(f1), "f2": float(f2), "T": float(T_act),
              "L": float(L), "level_dbfs": float(level_dbfs)}
    return Stimulus(name, x.astype(np.float32), params)


def sine(name: str, f: float, dur: float, level_dbfs: float) -> Stimulus:
    t = np.arange(int(round(dur * FS))) / FS
    x = _apply_fades(_amp(level_dbfs) * np.sin(2 * np.pi * f * t), 0.01, 0.01)
    params = {"kind": "sine", "f": float(f), "dur": float(dur), "level_dbfs": float(level_dbfs)}
    return Stimulus(name, x.astype(np.float32), params)


def stepped_sine(name: str, f: float, levels_dbfs: list[float], step_s: float = 1.0,
                 gap_s: float = 0.2) -> Stimulus:
    n_step, n_gap = int(round(step_s * FS)), int(round(gap_s * FS))
    t = np.arange(n_step) / FS
    unit = _apply_fades(np.sin(2 * np.pi * f * t), 0.01, 0.01)
    x = np.zeros(len(levels_dbfs) * (n_step + n_gap))
    starts = []
    for i, lv in enumerate(levels_dbfs):
        s = i * (n_step + n_gap)
        x[s:s + n_step] = _amp(lv) * unit
        starts.append(s)
    params = {"kind": "stepped_sine", "f": float(f), "levels_dbfs": [float(v) for v in levels_dbfs],
              "step_s": float(step_s), "gap_s": float(gap_s), "step_starts": starts}
    return Stimulus(name, x.astype(np.float32), params)


def ramp_sine(name: str, f: float, dur: float = 10.0, start_dbfs: float = -60.0,
              end_dbfs: float = -6.0) -> Stimulus:
    n = int(round(dur * FS))
    t = np.arange(n) / FS
    env_db = start_dbfs + (end_dbfs - start_dbfs) * t / dur
    x = _apply_fades(10.0 ** (env_db / 20.0) * np.sin(2 * np.pi * f * t), 0.01, 0.01)
    params = {"kind": "ramp_sine", "f": float(f), "dur": float(dur),
              "start_dbfs": float(start_dbfs), "end_dbfs": float(end_dbfs)}
    return Stimulus(name, x.astype(np.float32), params)


def bursts(name: str, f: float, level_dbfs: float, on_s: float = 0.2, off_s: float = 0.5,
           n: int = 8) -> Stimulus:
    n_on, n_off = int(round(on_s * FS)), int(round(off_s * FS))
    t = np.arange(n_on) / FS
    one = _apply_fades(_amp(level_dbfs) * np.sin(2 * np.pi * f * t), 0.001, 0.001)
    x = np.zeros(n * (n_on + n_off))
    starts = []
    for i in range(n):
        s = i * (n_on + n_off)
        x[s:s + n_on] = one
        starts.append(s)
    params = {"kind": "bursts", "f": float(f), "level_dbfs": float(level_dbfs), "on_s": float(on_s),
              "off_s": float(off_s), "n": int(n), "burst_starts": starts}
    return Stimulus(name, x.astype(np.float32), params)


def silence(name: str, dur: float) -> Stimulus:
    return Stimulus(name, np.zeros(int(round(dur * FS)), np.float32),
                    {"kind": "silence", "dur": float(dur)})


def level_steps(name: str, f: float, segments: list[tuple[float, float]]) -> Stimulus:
    """Continuous sine whose level jumps between `segments` [(level_dbfs, dur_s), ...]. Every
    duration is a whole number of periods, so each jump lands on a zero crossing (no click) and
    per-cycle analysis windows line up with the segment boundaries."""
    period = FS / f
    if period != int(period):
        raise ValueError(f"{f} Hz has no whole-sample period at {FS} Hz")
    lengths = [int(round(d * FS)) for _, d in segments]
    if any(n % int(period) for n in lengths):
        raise ValueError("every segment must last a whole number of periods")
    env = np.concatenate([np.full(n, _amp(lv)) for (lv, _), n in zip(segments, lengths)])
    t = np.arange(len(env)) / FS
    x = _apply_fades(env * np.sin(2 * np.pi * f * t), 0.01, 0.01)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    params = {"kind": "level_steps", "f": float(f),
              "segments": [{"start": int(s), "n": int(n), "level_dbfs": float(lv)}
                           for s, n, (lv, _) in zip(starts, lengths, segments)]}
    return Stimulus(name, x.astype(np.float32), params)


def decay_sine(name: str, f: float, start_dbfs: float, decay_db_per_s: float, dur: float) -> Stimulus:
    """Sine with a 2 ms attack and an exponential decay (linear in dB), like a plucked note."""
    t = np.arange(int(round(dur * FS))) / FS
    env = 10.0 ** ((start_dbfs - decay_db_per_s * t) / 20.0)
    x = _apply_fades(env * np.sin(2 * np.pi * f * t), 0.002, 0.01)
    params = {"kind": "decay_sine", "f": float(f), "start_dbfs": float(start_dbfs),
              "decay_db_per_s": float(decay_db_per_s), "dur": float(dur)}
    return Stimulus(name, x.astype(np.float32), params)


# Frequency-response sweeps shared by every pedal family and by calibration (C8 repeats sweep_m20).
# −50 dBFS (−35.5 dBu) is the small-signal level; −30 is already past the onset of a typical drive.
SWEEPS: dict[str, Stimulus] = {f"sweep_m{-lv}": sync_sweep(f"sweep_m{-lv}", 20, 20000, 10, lv)
                               for lv in (-50, -30, -20, -10)}
