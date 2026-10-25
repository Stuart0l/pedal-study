"""Ampero device discovery and synchronous play+record."""

import numpy as np
import sounddevice as sd

from .config import CLIP, DEVICE_MATCH, FS, IN_CH, OUT_CH, POST_S, PRE_S
from .signals import Stimulus


def _device_table() -> str:
    return "\n".join(f"  [{i}] {d['name']} (in {d['max_input_channels']}, out {d['max_output_channels']})"
                     for i, d in enumerate(sd.query_devices()))


def find_device() -> int:
    matches = [i for i, d in enumerate(sd.query_devices())
               if DEVICE_MATCH in d["name"] and d["max_input_channels"] >= 8
               and d["max_output_channels"] >= 4]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one '{DEVICE_MATCH}' device with >=8 in / >=4 out, "
                           f"found {len(matches)}:\n{_device_table()}")
    return matches[0]


def check_device(dev: int) -> None:
    try:
        sd.check_input_settings(device=dev, channels=8, samplerate=FS, dtype="float32")
        sd.check_output_settings(device=dev, channels=4, samplerate=FS, dtype="float32")
    except Exception as e:
        raise RuntimeError(f'{e}\nSet "Ampero II Stomp Audio" Format to {FS:,} Hz in Audio MIDI Setup') from e


def pad(stim: Stimulus) -> np.ndarray:
    return np.concatenate([np.zeros(int(round(PRE_S * FS)), np.float32), stim.data,
                           np.zeros(int(round(POST_S * FS)), np.float32)])


def capture(dev: int, stim: Stimulus, in_channels: list[int] | None = None) -> tuple[np.ndarray, bool]:
    """Play `stim` on USB Out OUT_CH, record `in_channels`. Returns (rec (N, ch), xrun)."""
    in_channels = in_channels or [IN_CH]
    out = pad(stim)[:, None]
    xrun = False
    for _ in range(2):
        rec = sd.playrec(out, samplerate=FS, device=dev, input_mapping=in_channels,
                         output_mapping=[OUT_CH], dtype="float32", latency="high")
        flags = sd.wait()
        xrun = bool(flags)
        if not xrun:
            break
    return np.asarray(rec, dtype=np.float32), xrun


def peak_dbfs(rec: np.ndarray) -> list[float]:
    pk = np.max(np.abs(rec), axis=0)
    return [float(20 * np.log10(max(p, 1e-12))) for p in pk]


def is_clipped(rec: np.ndarray) -> bool:
    return bool(np.any(np.abs(rec) >= CLIP))
