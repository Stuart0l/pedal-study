"""Compressor capture plan: test signals and steps T1-T3 (capture side; no analysis imports).

Unlike a drive, a compressor has memory (attack/release), so level-dependent tests hold each
level long enough to settle instead of ramping, and level changes are abrupt steps whose
settling is timed.
"""

from dataclasses import dataclass

from ..plan import Cap, PedalModel, Step, knob_dict
from ..signals import SWEEPS, Stimulus, decay_sine, level_steps, silence, stepped_sine

LO, HI = -40.0, -20.0   # attack/release jump: −25.5 ↔ −5.5 dBu, soft playing ↔ hard strum
# 1 kHz staircase: 3 dB steps, each held 2 s (the analysis reads the last 0.5 s, i.e. after
# up to 1.5 s of attack), then 0.2 s of silence.
STATIC_LEVELS = [float(v) for v in range(-60, -5, 3)]


def _stimuli() -> dict[str, Stimulus]:
    out = [SWEEPS[n] for n in ("sweep_m50", "sweep_m30", "sweep_m10")]
    out += [stepped_sine("comp_steps_1k", 1000, STATIC_LEVELS, step_s=2.0, gap_s=0.2),
            stepped_sine("comp_steps_100", 100, STATIC_LEVELS, step_s=2.0, gap_s=0.2),
            # three loud notes of 1 s between 2.5 s quiet stretches (first quiet part settles the start)
            level_steps("comp_attack_1k", 1000, [(LO, 2.0)] + [(HI, 1.0), (LO, 2.5)] * 3),
            # release after a short (0.1 s) and a long (2 s) loud note: program-dependent release?
            level_steps("comp_hold_1k", 1000, [(LO, 2.0), (HI, 0.1), (LO, 3.0), (HI, 2.0), (LO, 3.0)]),
            # plucked-note stand-in: −1.5 dBu at the attack, decaying 10 dB/s for 5 s
            decay_sine("comp_pluck_200", 200, -16.0, 10.0, 5.0),
            silence("silence_5s", 5.0)]
    return {s.name: s for s in out}


STIMULI: dict[str, Stimulus] = _stimuli()

Q = {
    "static": "C1 From which level does it compress, and how strongly?",
    "attack": "C2 How fast does it react and recover?",
    "hold": "C3 Does recovery depend on how long the loud part lasted?",
    "pluck": "C9 How does it change a decaying note?",
    "noise": "C7 How much noise does it add?",
    "bass": "C6 Distortion while compressing (100 Hz)",
    "tone": "C4/C5 Compression vs frequency; Tone shape and position",
    "V": "C8 Volume check: does Volume only shift the level?",
}


@dataclass(frozen=True)
class CompressorModel(PedalModel):
    """Volume, Comp (amount), Tone."""
    knobs: tuple[str, ...]           # panel order, as printed to the user
    comp_knob: str = "Comp"
    family = "compressor"
    notebook = "notebooks/03_compressor.ipynb"

    def steps(self) -> list[Step]:
        S = STIMULI
        c = self.comp_knob

        def _knobs(**changes) -> dict:
            return knob_dict(self.knobs, **changes)

        t1 = []
        for p in ("min", "noon", "max"):
            k = _knobs(**{c: p})
            t1 += [Cap(f"T1_steps_1k_comp-{p}", S["comp_steps_1k"], knobs=k, question=Q["static"]),
                   Cap(f"T1_attack_1k_comp-{p}", S["comp_attack_1k"], knobs=k, question=Q["attack"]),
                   Cap(f"T1_pluck_200_comp-{p}", S["comp_pluck_200"], knobs=k, question=Q["pluck"]),
                   Cap(f"T1_noise_comp-{p}", S["silence_5s"], knobs=k, question=Q["noise"], expect_silence=True)]
            if p == "noon":
                t1.append(Cap("T1_hold_1k_comp-noon", S["comp_hold_1k"], knobs=k, question=Q["hold"]))
            if p == "max":
                t1.append(Cap("T1_steps_100_comp-max", S["comp_steps_100"], knobs=k, question=Q["bass"]))
        return [
            Step("T1", "T1", f"{c} amount", f"at {c} min/noon/max: 1 kHz level staircase −60…−6 dBFS, "
                                            "20 dB level jumps, plucked-note decay, silence; plus release-hold "
                                            "test (noon) and 100 Hz staircase (max)",
                 "compression curve, attack/release, sustain, noise, bass distortion", t1),
            Step("T2", "T2", "Tone", "small-signal (−50), quiet (−30) and loud (−10) sweep at Tone min/noon/max",
                 "Tone's filter shape; whether it acts before the level detector; compression vs frequency",
                 [Cap(f"T2_sweep_{lv}_tone-{p}", S[f"sweep_{lv}"], knobs=_knobs(Tone=p), question=Q["tone"])
                  for p in ("min", "noon", "max") for lv in ("m50", "m30", "m10")]),
            Step("T3", "T3", "Volume", "quiet sweep at Volume 9:00 / noon / 3:00",
                 "confirm Volume only shifts the level",
                 [Cap(f"T3_sweep_m30_vol-{p}", S["sweep_m30"], knobs=_knobs(Volume=p), question=Q["V"])
                  for p in ("9", "noon", "3")]),
        ]


MODELS = {"compressor": CompressorModel(("Volume", "Comp", "Tone"))}
