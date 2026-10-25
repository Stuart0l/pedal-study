"""Rig constants. Channel numbers are 1-based, as used by sounddevice mappings.

Routing (Ampero II Stomp manual V2.5.0, p.65-66):
  USB Out 3/4 -> FX SEND (GLOBAL > USB Audio > FX SND SOURCE = USB OUT 3/4)
  USB In 7/8  <- INPUT L/R dry, before any processing
"""

from dataclasses import dataclass
from pathlib import Path

DEVICE_MATCH = "Ampero"          # substring of "Ampero II Stomp Audio"
FS = 48_000                      # at 96 kHz the Ampero (fw 2.5.0) streams USB In 5-8 as digital zero
OUT_CH = 3                       # USB Out 3 = FX SEND tip
IN_CH = 8                        # USB In 8 = INPUT R dry
ALL_IN = list(range(1, 9))
PRE_S = 0.5                      # silence before stimulus in every capture
POST_S = 1.0                     # silence after
CLIP = 0.999                     # |sample| >= CLIP => clipped
SEND_DBU_AT_0DBFS = 14.5         # manual spec: full-scale sine peak on USB out 3 -> +14.5 dBu RMS at send (unverified)
SESSIONS_DIR = Path("sessions")
# Part of the loopback gain G that lies AFTER the pedal (FX SEND jack -> INPUT R -> USB In 8).
# Measured 0 dB on 2026-10-24 (patch route: chain output on USB In 1 equalled the analog return),
# so G is attributed to the digital USB -> send side, i.e. it scales what the pedal receives.
INPUT_SIDE_GAIN_DB = 0.0


@dataclass(frozen=True)
class PedalModel:
    """Knob set of one pedal. T3 sweeps its `t3_knob` (the knob that is not Gain/Volume/Tone)."""
    knobs: tuple[str, ...]           # panel order, as printed to the user
    t3_knob: str
    t3_question: str
    t3_expect: str

    @property
    def t3_tag(self) -> str:         # capture-name tag: T3_sweep_<level>_<tag>-<pos>
        return self.t3_knob.lower()


# Every model needs Gain, Volume and Tone (T1, T2, T5 turn them).
PEDAL_MODELS = {
    "satcut": PedalModel(("Gain", "Volume", "Tone", "Cut"), "Cut",
                         "Q4 What is Saturation-Cut?",
                         "expected: post-clip low-pass whose corner moves"),
    "lowcut": PedalModel(("Volume", "Gain", "Tone", "Lowcut"), "Lowcut",
                         "Q4 What does Lowcut do, and is it before the clipper?",
                         "expected: high-pass; before the clipper it also tightens the bass distortion, "
                         "after it only thins the output"),
}
POS = {"min": "min (7:00)", "9": "9:00", "noon": "noon (12:00)", "3": "3:00", "max": "max (5:00)"}
