"""Capture-plan building blocks shared by calibration and every pedal family (capture side only)."""

from dataclasses import dataclass, field

from .config import FS, IN_CH
from .signals import Stimulus


@dataclass
class Cap:
    name: str
    stim: Stimulus
    in_channels: list[int] = field(default_factory=lambda: [IN_CH])
    knobs: dict | None = None        # full knob dict (pedal) or None (calibration)
    question: str = ""
    expect_clip: bool = False         # C5 top step may clip on purpose
    expect_silence: bool = False      # C3 noise, pedal noise captures


@dataclass
class Step:
    id: str
    key: str                          # --only name
    title: str
    desc: str
    why: str
    caps: list[Cap]

    @property
    def seconds(self) -> float:
        return sum(len(c.stim.data) / FS + 1.5 for c in self.caps)


def knob_dict(knobs: tuple[str, ...], **changes) -> dict:
    """All knobs at noon (in panel order) except `changes`."""
    return {**{k: "noon" for k in knobs}, **changes}


class PedalModel:
    """One pedal's knob set and test plan. Subclasses (one per family) set `family`,
    `notebook` and implement steps()."""
    family: str
    notebook: str
    knobs: tuple[str, ...]

    @property
    def defaults(self) -> dict:
        """Resting knob/switch positions (what "unchanged" means in the step list)."""
        return knob_dict(self.knobs)

    def steps(self) -> list[Step]:
        raise NotImplementedError
