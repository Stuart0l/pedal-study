"""Overdrive capture plan: test signals and steps T1-T5 (capture side; no analysis imports)."""

from dataclasses import dataclass

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
    "Q5": "Q5 Any dynamics (compression, sag, bias shift)?",
    "Q6": "Q6 Clean blend?",
    "V": "Volume check: does Volume only shift the level?",
}


@dataclass(frozen=True)
class OverdriveModel(PedalModel):
    """Gain, Volume, Tone plus one more control, which T3 tests: a knob (min/noon/max) or an on/off
    switch listed in `switches` (off by default; off/on, plus a 200 Hz ramp with it on, since a
    switch such as a boost may change the clipping itself)."""
    knobs: tuple[str, ...]           # panel order, as printed to the user
    t3_knob: str
    t3_question: str
    t3_expect: str
    switches: tuple[str, ...] = ()   # on/off controls; "off" unless a capture turns them on
    family = "overdrive"
    notebook = "notebooks/02_overdrive.ipynb"

    @property
    def t3_tag(self) -> str:         # capture-name tag: T3_sweep_<level>_<tag>-<pos>
        return self.t3_knob.lower()

    @property
    def t3_is_switch(self) -> bool:
        return self.t3_knob in self.switches

    @property
    def t3_positions(self) -> tuple[str, ...]:
        return ("off", "on") if self.t3_is_switch else ("min", "noon", "max")

    @property
    def t3_ref(self) -> str:         # position the others are compared with
        return "off" if self.t3_is_switch else "noon"

    def steps(self) -> list[Step]:
        S = STIMULI

        def _knobs(**changes) -> dict:
            return knob_dict(self.knobs, **{**{s: "off" for s in self.switches}, **changes})

        t1 = [Cap(f"T1_ramp_200_gain-{g}", S["ramp_200"], knobs=_knobs(Gain=g), question=f"{Q['Q1']} / {Q['Q2']}")
              for g in ("min", "noon", "max")]
        t1 += [Cap(f"T1_ramp_{f}", S[f"ramp_{f}"], knobs=_knobs(), question=f"{Q['Q1']} (vs frequency)")
               for f in ("100", "400", "1k", "3k")]

        def knob_sweeps(test, knob, tag, q, positions=("min", "noon", "max")):
            return [Cap(f"{test}_sweep_{lv}_{tag}-{p}", S[f"sweep_{lv}"], knobs=_knobs(**{knob: p}), question=q)
                    for p in positions for lv in ("m50", "m30", "m10")]

        t3 = knob_sweeps("T3", self.t3_knob, self.t3_tag, self.t3_question, self.t3_positions)
        if self.t3_is_switch:
            t3.append(Cap(f"T3_ramp_200_{self.t3_tag}-on", S["ramp_200"], knobs=_knobs(**{self.t3_knob: "on"}),
                          question=f"{self.t3_question} (clipping with it on)"))
        t3_desc = (f"small-signal (−50), quiet (−30) and loud (−10) sweep with {self.t3_knob} off/on, "
                   f"plus a 200 Hz ramp with it on" if self.t3_is_switch else
                   f"small-signal (−50), quiet (−30) and loud (−10) sweep at {self.t3_knob} min/noon/max")

        return [
            Step("T1", "T1", "Clip & Gain", "200 Hz ramp −60→−6 dBFS at Gain min/noon/max, "
                                            "then 100/400/1k/3k Hz ramps at noon",
                 "static transfer curve: clipping onset, knee shape, asymmetry; what Gain moves", t1),
            Step("T2", "T2", "Tone", "small-signal (−50), quiet (−30) and loud (−10) sweep at Tone min/noon/max",
                 "filter shapes; whether Tone changes how it distorts (pre-clip) or only re-EQs (post-clip)",
                 knob_sweeps("T2", "Tone", "tone", f"{Q['Q3']} / {Q['Q6']}")),
            Step("T3", "T3", self.t3_knob, t3_desc, self.t3_expect, t3),
            Step("T4", "T4", "Dynamics", "8 bursts of 200 Hz at −10 dBFS, 200 ms on / 500 ms off",
                 "compression, sag, blocking, bias shift",
                 [Cap("T4_bursts_200_m10", S["bursts_200_m10"], knobs=_knobs(), question=Q["Q5"])]),
            Step("T5", "T5", "Volume", "quiet sweep at Volume 9:00 / noon / 3:00",
                 "confirm Volume only shifts the level, then ignore it",
                 [Cap(f"T5_sweep_m30_vol-{p}", S["sweep_m30"], knobs=_knobs(Volume=p), question=Q["V"])
                  for p in ("9", "noon", "3")]),
        ]


MODELS = {
    "satcut": OverdriveModel(("Gain", "Volume", "Tone", "Cut"), "Cut",
                             "Q4 What is Saturation-Cut?",
                             "expected: post-clip low-pass whose corner moves"),
    "lowcut": OverdriveModel(("Volume", "Gain", "Tone", "Lowcut"), "Lowcut",
                             "Q4 What does Lowcut do, and is it before the clipper?",
                             "expected: high-pass; before the clipper it also tightens the bass distortion, "
                             "after it only thins the output"),
    "boost": OverdriveModel(("Volume", "Tone", "Gain"), "Boost",
                            "Q4 What does the Boost switch do: more gain into the clipper, or more level after it?",
                            "before the clipper: earlier clipping, ramp onset moves down; after it: same "
                            "clipping, louder output", switches=("Boost",)),
}
