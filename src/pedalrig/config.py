"""Rig constants. Channel numbers are 1-based, as used by sounddevice mappings.

Routing (Ampero II Stomp manual V2.5.0, p.65-66):
  USB Out 3/4 -> FX SEND (GLOBAL > USB Audio > FX SND SOURCE = USB OUT 3/4)
  USB In 7/8  <- INPUT L/R dry, before any processing
"""

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
KNOBS = ("Gain", "Volume", "Tone", "Cut")
POS = {"min": "min (7:00)", "9": "9:00", "noon": "noon (12:00)", "3": "3:00", "max": "max (5:00)"}
