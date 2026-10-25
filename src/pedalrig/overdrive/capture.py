"""Overdrive capture plan: test signals and steps T1-T5 (capture side; no analysis imports)."""

from dataclasses import dataclass, field
from typing import Literal

from ..plan import Cap, PedalModel, Step, knob_dict
from ..signals import SWEEPS, Stimulus, bursts, ramp_sine


def _stimuli() -> dict[str, Stimulus]:
    out = [SWEEPS[n] for n in ("sweep_m50", "sweep_m30", "sweep_m10")]
    out += [ramp_sine(f"ramp_{tag}", f) for tag, f in
            (("100", 100), ("200", 200), ("400", 400), ("1k", 1000), ("3k", 3000))]
    out.append(bursts("bursts_200_m10", 200, -10))
    return {s.name: s for s in out}


STIMULI: dict[str, Stimulus] = _stimuli()

Q = {
    "Q1": "Q1 Where does it clip, and how hard?",
    "Q2": "Q2 What does Gain change?",
    "Q3": "Q3 Is Tone before or after the clipper?",
    "Q4": {"freq": "Q4 What does the {c} control filter, and is it before the clipper?",
           "gain": "Q4 Does {c} add drive into the clipper or level after it, and does it change the EQ?"},
    "Q5": "Q5 Any dynamics (compression, sag, bias shift)?",
    "Q6": "Q6 Clean blend?",
    "V": "Volume check: does Volume only shift the level?",
}


KNOB_POSITIONS = ("min", "noon", "max")


@dataclass(frozen=True)
class OverdriveModel(PedalModel):
    """Gain, Volume, Tone and a fourth control `control` of category `kind` (see module doc)."""
    kind: Literal["freq", "gain"]
    control: str = "Fourth"
    positions: tuple[str, ...] = KNOB_POSITIONS   # gain: first = rest position
    family = "overdrive"
    notebook = "notebooks/02_overdrive.ipynb"

    knobs: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "knobs", ("Gain", "Volume", "Tone", self.control))

    @property
    def tag(self) -> str:            # capture-name tag: T3_sweep_<level>_<tag>-<pos>
        return "".join(ch for ch in self.control.lower() if ch.isalnum())

    @property
    def ref(self) -> str:            # position the others are compared with
        return "noon" if self.kind == "freq" else self.positions[0]

    @property
    def others(self) -> list[str]:
        return [p for p in self.positions if p != self.ref]

    @property
    def question(self) -> str:
        return Q["Q4"][self.kind].format(c=self.control)

    @property
    def defaults(self) -> dict:
        return knob_dict(self.knobs, **{self.control: self.ref})

    @property
    def session_meta(self) -> dict:
        return {"control": self.control, "positions": list(self.positions)}

    def steps(self) -> list[Step]:
        S = STIMULI

        def caps(test, knob, tag, positions, stims, q):
            return [Cap(f"{test}_{st}_{tag}-{p}", S[st], knobs={**self.defaults, knob: p}, question=q)
                    for p in positions for st in stims]

        sweeps = ("sweep_m50", "sweep_m30", "sweep_m10")
        t1 = caps("T1", "Gain", "gain", KNOB_POSITIONS, ("ramp_200",), f"{Q['Q1']} / {Q['Q2']}")
        t1 += [Cap(f"T1_ramp_{f}", S[f"ramp_{f}"], knobs=self.defaults, question=f"{Q['Q1']} (vs frequency)")
               for f in ("100", "400", "1k", "3k")]
        t3 = caps("T3", self.control, self.tag, self.positions, sweeps, self.question)
        t3_desc = f"small-signal (−50), quiet (−30) and loud (−10) sweep at {self.control} {'/'.join(self.positions)}"
        t3_why = "filter shape; same change at every level = after the clipper, flattened when loud = before it"
        if self.kind == "gain":
            t3 += caps("T3", self.control, self.tag, self.others, ("ramp_200",), f"{self.question} (clipping)")
            t3_desc += f", plus a 200 Hz ramp at {'/'.join(self.others)}"
            t3_why = "ramp vs the Gain-noon ramp: onset moves down = drive before the clipper, only louder = after; " + t3_why

        return [
            Step("T1", "T1", "Clip & Gain", "200 Hz ramp −60→−6 dBFS at Gain min/noon/max, "
                                            "then 100/400/1k/3k Hz ramps at noon",
                 "static transfer curve: clipping onset, knee shape, asymmetry; what Gain moves", t1),
            Step("T2", "T2", "Tone", "small-signal (−50), quiet (−30) and loud (−10) sweep at Tone min/noon/max",
                 "filter shapes; whether Tone changes how it distorts (pre-clip) or only re-EQs (post-clip)",
                 caps("T2", "Tone", "tone", KNOB_POSITIONS, sweeps, f"{Q['Q3']} / {Q['Q6']}")),
            Step("T3", "T3", f"{self.control} ({self.kind})", t3_desc, t3_why, t3),
            Step("T4", "T4", "Dynamics", "8 bursts of 200 Hz at −10 dBFS, 200 ms on / 500 ms off",
                 "compression, sag, blocking, bias shift",
                 [Cap("T4_bursts_200_m10", S["bursts_200_m10"], knobs=self.defaults, question=Q["Q5"])]),
            Step("T5", "T5", "Volume", "quiet sweep at Volume 9:00 / noon / 3:00",
                 "confirm Volume only shifts the level, then ignore it",
                 caps("T5", "Volume", "vol", ("9", "noon", "3"), ("sweep_m30",), Q["V"])),
        ]


MODELS = {"freq": OverdriveModel("freq"), "gain": OverdriveModel("gain", positions=("off", "on"))}
