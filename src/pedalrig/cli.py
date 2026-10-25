"""pedalrig CLI: capture only. Analysis lives in notebooks/ (pedalrig.analysis)."""

import argparse

CAL_KEYS = ["routing", "noise", "latency", "levels", "wideband", "refs", "repeat"]
PEDAL_KEYS = ["T1", "T2", "T3", "T4", "T5"]


def _devices() -> None:
    import sounddevice as sd

    from .audio import find_device

    try:
        match = find_device()
    except RuntimeError:
        match = None
    for i, d in enumerate(sd.query_devices()):
        mark = "*" if i == match else " "
        print(f"{mark} [{i}] {d['name']} (in {d['max_input_channels']}, out {d['max_output_channels']}, "
              f"{d['default_samplerate']:.0f} Hz)")
    if match is None:
        print("no unique Ampero device found")


def main() -> None:
    p = argparse.ArgumentParser(prog="pedalrig", description="Capture pedal measurements via the Ampero II Stomp.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices", help="list audio devices, mark the Ampero")

    c = sub.add_parser("calibrate", help="cable loopback calibration: FX SEND -> INPUT R")
    c.add_argument("--label", default="cable")
    c.add_argument("--only", nargs="+", choices=CAL_KEYS, help="run only these steps (session not complete)")
    c.add_argument("--list", action="store_true", help="print the steps and exit")
    c.add_argument("--yes", action="store_true", help="no prompts; keep captures with warnings")

    q = sub.add_parser("pedal", help="pedal tests T1-T5: FX SEND -> pedal -> INPUT R")
    q.add_argument("--label", default="pedal")
    q.add_argument("--calibration", help="calibration session dir (default: newest complete)")
    q.add_argument("--only", nargs="+", choices=PEDAL_KEYS)
    q.add_argument("--list", action="store_true")
    q.add_argument("--yes", action="store_true")

    a = p.parse_args()
    if a.cmd == "devices":
        _devices()
        return

    from .steps import calibration_steps, pedal_steps, print_list, run

    kind = "calibration" if a.cmd == "calibrate" else "pedal"
    if a.list:
        print_list(calibration_steps() if kind == "calibration" else pedal_steps(), kind)
        return
    run(kind, a.label, a.only, a.yes, getattr(a, "calibration", None))


if __name__ == "__main__":
    main()
