"""Registry of measurable pedals across families (capture side; no analysis imports)."""

import dataclasses
import json
from pathlib import Path

import numpy as np

from .compressor import capture as compressor
from .config import SESSIONS_DIR
from .overdrive import capture as overdrive
from .plan import PedalModel
from .session import list_sessions
from .signals import SWEEPS, Stimulus

PEDAL_MODELS: dict[str, PedalModel] = {**overdrive.MODELS, **compressor.MODELS}


def _all_stimuli() -> dict[str, Stimulus]:
    """Every pedal-test signal of every family, for the calibration reference loopbacks."""
    out: dict[str, Stimulus] = {}
    for stims in (SWEEPS, overdrive.STIMULI, compressor.STIMULI):
        for name, s in stims.items():
            if name in out and not np.array_equal(out[name].data, s.data):
                raise ValueError(f"two different signals named {name}")
            out[name] = s
    return out


ALL_PEDAL_STIMULI = _all_stimuli()


def build_model(name: str, control: str | None = None, positions: list[str] | tuple[str, ...] | None = None) -> PedalModel:
    """PEDAL_MODELS[name] with an overdrive's fourth-control name and positions filled in."""
    pm = PEDAL_MODELS[name]
    if isinstance(pm, overdrive.OverdriveModel) and control:
        pm = dataclasses.replace(pm, control=control, positions=tuple(positions or pm.positions))
    return pm


def model_of(session_path: Path) -> PedalModel:
    meta = json.loads((Path(session_path) / "meta.json").read_text())
    return build_model(meta["pedal_model"], meta.get("control"), meta.get("positions"))


def latest_pedal_session(family: str, root: Path = SESSIONS_DIR) -> Path:
    """Newest pedal session whose model belongs to `family` ("overdrive" or "compressor")."""
    for p in reversed(list_sessions("pedal", root)):
        if model_of(p).family == family:
            return p
    raise FileNotFoundError(f"no {family} pedal session: run uv run pedalrig pedal --model <{family} model>")
