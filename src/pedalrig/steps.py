"""Measurement step definitions and the interactive capture runner.

Capture only: prints what to set, plays/records, checks peak/clip/xrun/dropouts, saves WAVs + meta.json.
All analysis lives in analysis.py / the notebooks.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd

from .audio import capture, check_device, find_device, is_clipped, peak_dbfs
from .config import ALL_IN, FS, IN_CH, PEDAL_MODELS, POS, PedalModel
from .session import Session, latest_calibration
from .signals import PEDAL_STIMULI, Stimulus, silence, sine, stepped_sine, sync_sweep

CAL_CHECKLIST = """\
Before calibrating (cable loopback):
 1. One TS instrument cable: Ampero FX SEND (back) -> INPUT R. Nothing in INPUT L. Phones volume low.
 2. GLOBAL -> USB Audio: FX SND SOURCE = USB OUT OFF, OUTPUT SOURCE = USB OUT 1/2,
    REC LEVEL(7/8) = 0dB (leave it there for good), MONITOR LEVEL = 0dB.
 3. GLOBAL -> I/O: FX SND SOURCE = Normal, OUTPUT SOURCE = USB Only, INPUT MODE(R) = E.GT.
 4. Saved send patch, all slots empty: chain A input SOURCE = USB 3/4, INPUT CH = L, LEVEL = 0dB;
    chain A output OUTPUT TO = FX SND, OUT LEVEL = 0dB (the direct USB OUT 3/4 -> FX SND route
    loses 6 dB; this patch gives a 0 dB loop). Unit NOT bypassed.
 5. macOS: Audio MIDI Setup -> Ampero II Stomp Audio = 48,000 Hz (USB In 5-8 are silent at 96 kHz),
    and the Ampero's main volume at maximum (0 dB; a lower slider scales every stimulus).
    Allow the terminal app microphone access if macOS prompts."""

PEDAL_CHECKLIST = """\
Before the pedal run:
 1. Ampero FX SEND -> pedal IN, pedal OUT -> Ampero INPUT R.
 2. Pedal powered from a regulated supply (not a battery). Pedal ON (LED lit).
 3. Ampero settings unchanged since calibration (REC LEVEL(7/8) still 0dB).
 4. Knob positions marked (tape/photo) so every setting is repeatable."""

ROUTING_FIX = ("check the send patch (chain A: SOURCE = USB 3/4 L, LEVEL = 0dB, OUTPUT TO = FX SND, OUT LEVEL = 0dB), "
               "GLOBAL→I/O FX SND SOURCE = Normal, GLOBAL→USB Audio FX SND SOURCE = USB OUT OFF; unit not bypassed")


@dataclass
class Cap:
    name: str
    stim: Stimulus
    in_channels: list[int] = field(default_factory=lambda: [IN_CH])
    knobs: dict | None = None        # full knob dict (pedal) or None (calibration)
    question: str = ""
    expect_clip: bool = False         # C5 top step may clip on purpose
    expect_silence: bool = False      # C3 noise


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


class Quit(Exception):
    pass


class Prompter:
    def __init__(self, yes: bool):
        self.yes = yes

    def choose(self, prompt: str, choices: str, default: str) -> str:
        """Return one char of `choices`; Enter -> default. --yes -> default."""
        if self.yes:
            return default
        while True:
            ans = input(f"  {prompt} ").strip().lower()
            if ans == "":
                return default
            if ans[0] in choices:
                return ans[0]


# ---------------------------------------------------------------- definitions

def calibration_steps() -> list[Step]:
    lat = sync_sweep("lat", 100, 20000, 1.0, -20)
    sweep20 = PEDAL_STIMULI["sweep_m20"]
    return [
        Step("C2", "routing", "Routing", "1 kHz sine −20 dBFS, 2 s, record all 8 USB inputs",
             "proves USB Out 3 reaches FX SEND and the cable lands on INPUT R (USB In 8)",
             [Cap("routing_1k", sine("routing_1k", 1000, 2.0, -20), in_channels=ALL_IN)]),
        Step("C3", "noise", "Noise floor", "5 s of silence",
             "residual noise, DC and mains hum of the loop",
             [Cap("noise", silence("noise", 5.0), expect_silence=True)]),
        Step("C4", "latency", "Latency & polarity", "5 × 1 s sweep 100 Hz–20 kHz at −20 dBFS",
             "round-trip delay in samples and whether it is identical every capture",
             [Cap(f"latency_{i}", lat) for i in range(1, 6)]),
        Step("C5", "levels", "Levels/linearity", "1 kHz sine, 9 steps −60…−1 dBFS",
             "loopback gain, linearity and clip point of the interface path",
             [Cap("steps_1k", stepped_sine("steps_1k", 1000, [-60, -50, -40, -30, -20, -12, -6, -3, -1]),
                  expect_clip=True)]),
        Step("C6", "wideband", "Wideband response", "sweep 10 Hz–23 kHz at −20 dBFS, 10 s",
             "frequency response and bandwidth of the loop itself (internal DSP is 44.1 kHz)",
             [Cap("wide_sweep", sync_sweep("wide", 10, 23000, 10, -20))]),
        Step("C7", "refs", "Reference loopbacks", "every pedal-test signal once through the cable",
             "per-signal reference that pedal captures are divided by",
             [Cap(f"ref_{n}", s) for n, s in PEDAL_STIMULI.items()]),
        Step("C8", "repeat", "Repeatability", "sweep_m20 twice more",
             "how much two identical captures differ (measurement noise floor)",
             [Cap(f"repeat_sweep_m20_{i}", sweep20) for i in (2, 3)]),
    ]


Q = {
    "Q1": "Q1 Where does it clip, and how hard?",
    "Q2": "Q2 What does Gain change?",
    "Q3": "Q3 Is Tone before or after the clipper?",
    "Q5": "Q5 Any dynamics (compression, sag, bias shift)?",
    "Q6": "Q6 Clean blend?",
    "V": "Volume check: does Volume only shift the level?",
}


def pedal_steps(model: PedalModel) -> list[Step]:
    S = PEDAL_STIMULI

    def _knobs(**changes) -> dict:
        return {**{k: "noon" for k in model.knobs}, **changes}

    t1 = [Cap(f"T1_ramp_200_gain-{g}", S["ramp_200"], knobs=_knobs(Gain=g), question=f"{Q['Q1']} / {Q['Q2']}")
          for g in ("min", "noon", "max")]
    t1 += [Cap(f"T1_ramp_{f}", S[f"ramp_{f}"], knobs=_knobs(), question=f"{Q['Q1']} (vs frequency)")
           for f in ("100", "400", "1k", "3k")]

    def knob_sweeps(test, knob, tag, q):
        return [Cap(f"{test}_sweep_{lv}_{tag}-{p}", S[f"sweep_{lv}"], knobs=_knobs(**{knob: p}), question=q)
                for p in ("min", "noon", "max") for lv in ("m50", "m30", "m10")]

    return [
        Step("T1", "T1", "Clip & Gain", "200 Hz ramp −60→−6 dBFS at Gain min/noon/max, "
                                        "then 100/400/1k/3k Hz ramps at noon",
             "static transfer curve: clipping onset, knee shape, asymmetry; what Gain moves", t1),
        Step("T2", "T2", "Tone", "small-signal (−50), quiet (−30) and loud (−10) sweep at Tone min/noon/max",
             "filter shapes; whether Tone changes how it distorts (pre-clip) or only re-EQs (post-clip)",
             knob_sweeps("T2", "Tone", "tone", f"{Q['Q3']} / {Q['Q6']}")),
        Step("T3", "T3", model.t3_knob,
             f"small-signal (−50), quiet (−30) and loud (−10) sweep at {model.t3_knob} min/noon/max",
             model.t3_expect, knob_sweeps("T3", model.t3_knob, model.t3_tag, model.t3_question)),
        Step("T4", "T4", "Dynamics", "8 bursts of 200 Hz at −10 dBFS, 200 ms on / 500 ms off",
             "compression, sag, blocking, bias shift",
             [Cap("T4_bursts_200_m10", S["bursts_200_m10"], knobs=_knobs(), question=Q["Q5"])]),
        Step("T5", "T5", "Volume", "quiet sweep at Volume 9:00 / noon / 3:00",
             "confirm Volume only shifts the level, then ignore it",
             [Cap(f"T5_sweep_m30_vol-{p}", S["sweep_m30"], knobs=_knobs(Volume=p), question=Q["V"])
              for p in ("9", "noon", "3")]),
    ]


def print_list(steps: list[Step], kind: str) -> None:
    print(f"{kind} steps:")
    if kind == "calibration":
        print("  C0  Checklist      Ampero settings + cabling (always)")
        print(f"  C1  Device         find Ampero, check {FS // 1000} kHz duplex (always)")
    for s in steps:
        print(f"  {s.id:<3} {s.title:<20} {s.desc}  (≈{s.seconds:.0f} s, --only {s.key})")
        print(f"      why: {s.why}")
        if kind == "pedal":
            for c in s.caps:
                changed = {k: v for k, v in c.knobs.items() if v != "noon"}
                print(f"        - {c.name:<26} {changed or 'all noon'}")
    total = sum(s.seconds for s in steps)
    print(f"  total capture time ≈ {total / 60:.1f} min plus knob changes")


# ---------------------------------------------------------------- runner

def _device_info(dev: int) -> dict:
    d = sd.query_devices(dev)
    print(f"  device [{dev}] {d['name']}: in {d['max_input_channels']}, out {d['max_output_channels']}, "
          f"high latency in {d['default_high_input_latency'] * 1e3:.1f} ms / "
          f"out {d['default_high_output_latency'] * 1e3:.1f} ms")
    return {"name": d["name"], "index": dev}


def _open_device() -> dict:
    print("\n[C1] Device")
    try:
        dev = find_device()
        check_device(dev)
    except RuntimeError as e:
        print(f"  FAIL: {e}")
        sys.exit(1)
    # Right after the Ampero is (re)connected, its first stream delivered ~1 s of digital zeros
    # on every input; a discarded 2 s silent capture absorbs that before real measurements.
    capture(dev, silence("warmup", 2.0))
    return _device_info(dev)


def _routing_hint(rec) -> str | None:
    pk = peak_dbfs(rec)  # index i -> USB In i+1
    print("  per-channel peak dBFS: " + "  ".join(f"ch{i + 1} {p:6.1f}" for i, p in enumerate(pk)))
    ch7, ch8 = pk[6], pk[7]
    if max(pk) < -120:
        return "macOS blocks the mic: System Settings → Privacy & Security → Microphone → enable your terminal app"
    if ch7 > ch8:
        return "cable is in INPUT L; move it to INPUT R"
    if ch8 - ch7 < 20:
        return ROUTING_FIX
    print(f"  loop gain ≈ {ch8 + 20:+.1f} dB (peak re −20 dBFS stimulus)")
    return None


def _health(cap: Cap, rec, xrun: bool, pedal: bool) -> list[str]:
    col = cap.in_channels.index(IN_CH)
    pk = peak_dbfs(rec)
    clipped = is_clipped(rec[:, [col]])
    print(f"  peak {pk[col]:.1f} dBFS (USB In {IN_CH})" + ("  CLIPPED" if clipped else "") + ("  XRUN" if xrun else ""))
    problems = []
    if clipped:
        if cap.expect_clip:
            print("  (clipping expected near full scale)")
        else:
            problems.append("input clipped" + (": lower pedal Volume" if pedal else ""))
    if not cap.expect_silence and pk[col] < -80:
        problems.append("no signal" + (": pedal on? cables?" if pedal else ""))
    if xrun:
        problems.append("buffer over/underrun (xrun) even after automatic retry")
    dead = _dead_seconds(rec[:, col])
    if dead > 0:
        problems.append(f"input went digitally silent for {dead:.2f} s (below any analog noise floor): "
                        "was an Ampero setting (REC/MONITOR LEVEL, routing) touched during the capture?")
    return problems


def _dead_seconds(x, block_s: float = 0.05, floor_dbfs: float = -135.0) -> float:
    """Seconds of 50 ms blocks quieter than any connected analog input can be (muted/dropped)."""
    n = int(block_s * FS)
    nb = len(x) // n
    rms = np.sqrt(np.mean(x[:nb * n].reshape(nb, n).astype(np.float64) ** 2, axis=1))
    return float(np.sum(rms < 10 ** (floor_dbfs / 20)) * block_s)


def _run_capture(sess: Session, dev: int, step: Step, cap: Cap, ui: Prompter, pedal: bool) -> None:
    sess.save_stimulus(cap.stim)
    while True:
        print(f"  capturing {cap.name} ({len(cap.stim.data) / FS + 1.5:.1f} s)...", flush=True)
        rec, xrun = capture(dev, cap.stim, cap.in_channels)
        if step.id == "C2":
            hint = _routing_hint(rec)
            sess.save_capture(cap.name, cap.stim.name, step.id, rec, cap.in_channels, cap.knobs, xrun, False)
            if hint:
                print(f"  ROUTING PROBLEM: {hint}")
                if ui.yes or ui.choose("r = retake, q = quit:", "rq", "r") == "q":
                    raise SystemExit(1)
                continue
        problems = _health(cap, rec, xrun, pedal)
        kept = False
        if problems:
            for p in problems:
                print(f"  WARNING: {p}")
            ans = ui.choose("r = retake, k = keep and continue, q = quit:", "rkq", "k")
            if ans == "r":
                continue
            if ans == "q":
                raise Quit
            kept = True
        sess.save_capture(cap.name, cap.stim.name, step.id, rec, cap.in_channels, cap.knobs, xrun, kept)
        return


def _print_knobs(knobs: dict, prev: dict | None) -> None:
    for k in knobs:  # panel order (pedal_steps builds every dict from model.knobs)
        mark = "→" if prev is None or prev[k] != knobs[k] else " "
        print(f"    {mark} {k:<7} {POS[knobs[k]]}")


def run(kind: str, label: str, only: list[str] | None, yes: bool, calibration: str | None = None,
        model: str | None = None) -> Path:
    """`model` (a PEDAL_MODELS key) is required for kind == "pedal"."""
    ui = Prompter(yes)
    pedal = kind == "pedal"
    steps = pedal_steps(PEDAL_MODELS[model]) if pedal else calibration_steps()
    if only:
        keys = {s.key for s in steps}
        bad = [o for o in only if o not in keys]
        if bad:
            raise SystemExit(f"unknown --only {bad}; choose from {sorted(keys)}")
        steps = [s for s in steps if s.key in only]

    cal_ref = None
    if pedal:
        try:
            cal_ref = Path(calibration) if calibration else latest_calibration()
        except FileNotFoundError as e:
            raise SystemExit(str(e))
        if not (cal_ref / "meta.json").is_file():
            raise SystemExit(f"not a session directory: {cal_ref}")

    print(f"\n[{'P0' if pedal else 'C0'}] Checklist\n{PEDAL_CHECKLIST if pedal else CAL_CHECKLIST}")
    if not yes:
        input("  Enter when done (Ctrl-C to abort) ")
    device = _open_device()
    sess = Session.create(kind, label, device)
    sess.meta["checklist_confirmed"] = True
    sess.meta["calibration_ref"] = str(cal_ref) if cal_ref else None
    if pedal:
        sess.meta["pedal_model"] = model
    sess.save_meta()
    print(f"  session: {sess.path}")
    if cal_ref:
        print(f"  calibration: {cal_ref}")

    complete = only is None
    prev_knobs = None
    last_id = "T5" if pedal else "C8"
    try:
        for step in steps:
            print(f"\n[{step.id}/{last_id}] {step.title} — {step.desc} (≈{step.seconds:.0f} s)")
            print(f"  Why: {step.why}")
            if not pedal:
                ans = ui.choose("Enter = run, s = skip, q = quit:", "sq", "r")
                if ans == "q":
                    raise Quit
                if ans == "s":
                    complete = False
                    continue
            for i, cap in enumerate(step.caps, 1):
                if pedal:
                    print(f"\n  ({i}/{len(step.caps)}) {cap.name}   {cap.question}")
                    if cap.knobs == prev_knobs:
                        # nothing to turn: capture straight away (Ctrl-C still stops the run)
                        print("    knobs unchanged, capturing")
                    else:
                        print("    knobs (→ = turn now):")
                        _print_knobs(cap.knobs, prev_knobs)
                        ans = ui.choose("Enter = run, s = skip, q = quit:", "sq", "r")
                        if ans == "q":
                            raise Quit
                        if ans == "s":
                            complete = False
                            continue
                        prev_knobs = cap.knobs
                _run_capture(sess, device["index"], step, cap, ui, pedal)
    except (Quit, KeyboardInterrupt):
        print("\n  stopped; captured data so far is saved.")
        complete = False

    sess.meta["complete"] = complete
    sess.save_meta()
    nb = "notebooks/02_pedal.ipynb" if pedal else "notebooks/01_calibration.ipynb"
    print(f"\nsaved {len(sess.meta['captures'])} captures to {sess.path} (complete={complete})")
    print(f"analyze in {nb}")
    return sess.path
