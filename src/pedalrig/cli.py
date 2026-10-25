"""pedalrig CLI: capture only. Analysis lives in notebooks/ (pedalrig.analysis)."""

import argparse

from .models import PEDAL_MODELS, build_model

CAL_KEYS = ["routing", "noise", "latency", "levels", "wideband", "refs", "repeat"]


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

    q = sub.add_parser("pedal", help="pedal tests: FX SEND -> pedal -> INPUT R")
    q.add_argument("--model", required=True, choices=sorted(PEDAL_MODELS),
                   help="freq / gain = overdrive whose fourth control filters / changes level or drive; "
                        "compressor")
    q.add_argument("--control", help='overdrive: panel name of the fourth control, e.g. Cut, Lowcut, Boost, "LP/HP"')
    q.add_argument("--positions", help="gain: comma-separated switch positions, rest first (default off,on)")
    q.add_argument("--label", help="session label (default: the model name)")
    q.add_argument("--calibration", help="calibration session dir (default: newest complete)")
    q.add_argument("--only", nargs="+", help="run only these tests (ids from --list); session not complete")
    q.add_argument("--list", action="store_true")
    q.add_argument("--yes", action="store_true")

    a = p.parse_args()
    if a.cmd == "devices":
        _devices()
        return

    from .steps import calibration_steps, print_list, run

    kind = "calibration" if a.cmd == "calibrate" else "pedal"
    if kind == "calibration":
        if a.list:
            print_list(calibration_steps(), kind)
        else:
            run(kind, a.label, a.only, a.yes)
        return
    pm = build_model(a.model, a.control, a.positions and a.positions.split(","))
    if pm.session_meta and not a.control:
        q.error(f"--model {a.model} needs --control <panel name of the fourth control>")
    if a.list:
        print_list(pm.steps(), kind, pm.defaults)
    else:
        run(kind, a.label or a.model, a.only, a.yes, a.calibration, a.model, pm)


if __name__ == "__main__":
    main()
