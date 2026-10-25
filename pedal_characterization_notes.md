# Characterizing a guitar drive pedal by black-box measurement

Notes compiled from a conversation on 2026-10-23.

**Subject:** a drive pedal with four knobs — Gain, Volume, Tone, Saturation-Cut. No schematic available.
**Goal (current stage):** learn its characteristics to use it better in an effects chain. Modeling is not the aim yet.
**Available hardware:** Hotone Ampero II Stomp (firmware 2.5.0), which works as an 8-in/8-out USB audio interface.

---

## 1. Measurement approach

Treat the pedal as a black box: feed it known signals, record the output, infer structure.

### Generic rig (any audio interface)

```
DAW/computer → interface line out → reamp box → pedal → interface Hi-Z input → DAW
                                      └── loopback (pedal removed) ──┘
```

- A reamp box (or a passive DI run backwards) converts line level to instrument level and presents the high-impedance source the pedal expects. Without it you measure "pedal plus wrong impedance."
- Record a loopback with the pedal removed, same settings, for every test signal. It gives the system's own response and the exact latency offset.
- 48 or 96 kHz, 24-bit. Distortion creates harmonics well above the input.
- Keep the interface input far from clipping; set its gain once and never touch it.
- Short cables, same cables for every capture. Regulated power supply, not a battery.

### Using the Ampero II Stomp instead (from the manual)

The FX SEND jack is an instrument-level, 1 kΩ, unbalanced stereo TRS output (tip = left), nominal −17.78 dBu, max +14.5 dBu. Inputs L/R are tapped directly to USB In 7/8 before any processing, so INPUT R is a clean capture point.

Signal path:

```
DAW track 1 → USB out 3/4 → FX SEND → pedal → INPUT R → USB in 8 → DAW track 2
```

Ampero settings:

1. Global → USB Audio: `FX SND SOURCE = USB OUT 3/4` (default is off). Leave `OUTPUT SOURCE = USB OUT 1/2` for headphone monitoring.
2. Global → I/O: `FX SND SOURCE = USB Only` so no patch effects reach the send. `INPUT MODE (R) = E.GT` (1 MΩ). Line mode is 10 kΩ and would load the pedal's output.
3. Load an empty patch with no FX SND / FX LOOP modules. Do not engage bypass — bypass disables USB audio.
4. Sample rate 48 kHz via the Hotone ASIO panel (Windows) or Audio MIDI Setup (Mac). If only 44.1 kHz is offered, the USB firmware needs updating (V2.01+).
5. Global → USB Audio → `REC LEVEL (7/8)`: set so the pedal at max Gain and max Volume peaks around −6 dBFS, then leave it.

DAW: track 1 input None, output USB 3/4, holds the test WAVs (signal on the left channel). Track 2 input USB In 8 mono, output 1/2, record-armed, software monitoring off.

Cabling: a plain TS instrument cable from FX SEND picks up the left channel and grounds the ring; that is fine.

Level reference: 0 dBFS on USB out 3 ≈ +14.5 dBu at the send jack, so −20 dBFS ≈ −5 dBu (roughly a hard-strummed humbucker). A sweep from −40 to about −6 dBFS covers everything a guitar produces.

Source impedance caveat: a guitar presents 6–15 kΩ plus inductance; the send presents 1 kΩ. For buffered/op-amp-input drives this is inaudible. For fuzzes or anything that responds to the guitar volume knob, add a 10 kΩ series resistor on the cable into the pedal, or note that results describe the pedal "after a buffer."

### Ampero Sound Clone (alternative quick path)

Global → Sound Clone: FX SEND → pedal → INPUT R, `TARGET TYPE = Distortion`, `CAB INCLUDED = No`, `INPUT MODE (R) = 1 MΩ`, adjust `TARGET LEVEL`, press CLONE. Produces a CLONE module and an A/B comparison screen. The manual states it cannot clone drive pedals that blend in clean signal (Tube Screamer style). Useful as a listening reference, but it reveals nothing about *why* the pedal sounds as it does.

---

## 2. Test signals

| Signal | Purpose |
|---|---|
| Exponential sine sweep, 20 Hz–20 kHz, ~10 s | Frequency response; via the Farina method, separate impulse responses per harmonic order (distortion vs. frequency) |
| Stepped/ramped sine at ~100, 200, 400, 1k, 3k Hz, amplitude ramp silence→max | Static transfer curve: onset of clipping, soft vs. hard knee, symmetric vs. asymmetric |
| Tone bursts (200 ms on, 500 ms off) | Time-dependent behaviour: compression, sag, blocking distortion, bias shift |
| Real guitar DI (only if modeling later) | Training material and listening tests |

Play each at several levels, e.g. −30, −20, −10, 0 dB relative to a hard-strummed humbucker.

---

## 3. Characterization pass (no modeling)

About 15 captures of ~10 s each. Each answers one question.

1. **Where does it clip, and how hard?** 200 Hz sine ramp −40 → −6 dBFS over 10 s, all knobs at noon. Plot output sample vs. input sample. Straight = clean; the bend = onset; gentle bend = soft clipping; sharp knee = hard clipping; top/bottom halves differ = asymmetric (more even harmonics). Repeat at three Gain positions.
2. **What does Gain change?** Same ramp at Gain min/noon/max. Only the onset moves → pre-clip boost. Curve shape changes → multiple stages or diode switching. Note how output level changes with Gain.
3. **Is Tone before or after the clipper?** Quiet sweep (−30 dBFS) at Tone min/noon/max gives the filter shapes. Loud sweep (−10 dBFS): if Tone changes *how* it distorts, it is pre-clip; if it only re-EQs the same grit, it is post-clip. Pre-clip tone controls affect feel and pick response.
4. **What is Saturation-Cut?** Same test. Expected: a post-clip low-pass whose corner moves with the knob and no change to the clipping curve. If it alters the clipping curve or the bass entering the clipper, it is something else.
5. **Any dynamics?** Bursts of 200 Hz at −10 dBFS. Flat envelope = memoryless. Sag/overshoot = compression or supply sag. Notes that start clean then dirty (or vice versa) = bias shift.
6. **Clean blend?** Compare low end of loud vs. quiet sweeps. Clean-blend designs keep bass tight; straight clippers distort everything equally.

Verify Volume once: at three settings with a quiet sweep, confirm only the level shifts, then fix it and ignore it.

---

## 4. Reading the results for chain placement

- **Clipping onset in dBu** says how hot a signal it wants. Low onset → saturates on everything; place after a volume control or compressor. High onset → wants a boost in front.
- **Pre-clip Tone** → EQ and boosts *before* the pedal matter a lot. **Post-clip Tone** → EQ after it does the same job.
- **Post-clip low-pass (Cut)** does cab-like work; with a cab sim or amp after it, Cut can likely stay open.
- **Output level** at preferred settings determines how it stacks into the next pedal's input.

---

## 5. If modeling later

- For understanding: fit a Wiener–Hammerstein model (input filter → static nonlinearity → output filter); the blocks map onto the knobs.
- For a usable plugin: Neural Amp Modeler or GuitarML Proteus (snapshot per knob setting); AIDA-X or GuitarML's conditioned LSTM for knob-conditioned models from a grid (e.g. Gain × Tone × Cut at 5 positions = 125 captures).
- Knob positions must be marked (tape/photos); repeatability is the largest error source.

## 6. Offered next step

A Python script to generate the test WAVs, and another to analyze the recordings (transfer-curve plots, frequency responses, harmonic breakdown, burst envelopes).
