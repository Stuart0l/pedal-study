"""On-disk session format.

sessions/stimuli/<name>-<sha1[:12]>.wav     shared test signals (float32), one file per unique content
sessions/<YYYYmmdd-HHMMSS>_<kind>_<label>/
    meta.json                               stimulus "file" paths are relative to the session dir
    captures/<name>.flac                    24-bit FLAC: bit-exact for the Ampero's 24-bit input
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import is_clipped, peak_dbfs
from .config import FS, IN_CH, OUT_CH, POST_S, PRE_S, SESSIONS_DIR
from .signals import Stimulus


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Session:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())

    @classmethod
    def create(cls, kind: str, label: str, device: dict | None = None,
               root: Path = SESSIONS_DIR) -> "Session":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(root) / f"{stamp}_{kind}_{label}"
        path.mkdir(parents=True)
        (path / "captures").mkdir()
        meta = {"kind": kind, "label": label, "created": _now(), "fs": FS,
                "device": {"name": None, "index": None, "out_channel": OUT_CH, "in_channel": IN_CH,
                           **(device or {})},
                "pre_s": PRE_S, "post_s": POST_S, "checklist_confirmed": False,
                "calibration_ref": None, "complete": False, "stimuli": {}, "captures": []}
        (path / "meta.json").write_text(json.dumps(meta, indent=1))
        return cls(path)

    def save_meta(self) -> None:
        tmp = self.path / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=1))
        tmp.replace(self.path / "meta.json")

    def save_stimulus(self, stim: Stimulus) -> None:
        """Store the signal once in the shared sessions/stimuli/ store, keyed by content hash,
        so identical signals across sessions share one file and a changed signal never
        overwrites an old one."""
        if stim.name in self.meta["stimuli"]:
            return
        data = np.ascontiguousarray(stim.data, dtype=np.float32)
        digest = hashlib.sha1(data.tobytes() + str(FS).encode()).hexdigest()[:12]
        store = self.path.parent / "stimuli"
        store.mkdir(exist_ok=True)
        target = store / f"{stim.name}-{digest}.wav"
        if not target.exists():
            tmp = store / f".{target.name}.tmp"
            sf.write(tmp, data, FS, subtype="FLOAT", format="WAV")
            tmp.replace(target)
        self.meta["stimuli"][stim.name] = {**stim.params, "file": f"../stimuli/{target.name}"}
        self.save_meta()

    def save_capture(self, name: str, stim_name: str, step: str, rec: np.ndarray,
                     in_channels: list[int], knobs: dict | None, xrun: bool,
                     kept_with_warning: bool) -> None:
        rel = f"captures/{name}.flac"
        sf.write(self.path / rel, rec, FS, format="FLAC", subtype="PCM_24")
        entry = {"name": name, "stimulus": stim_name, "step": step, "knobs": knobs,
                 "in_channels": list(in_channels), "file": rel, "peak_dbfs": peak_dbfs(rec),
                 "clipped": is_clipped(rec), "xrun": xrun, "kept_with_warning": kept_with_warning,
                 "time": _now()}
        caps = [c for c in self.meta["captures"] if c["name"] != name]  # retake replaces
        caps.append(entry)
        self.meta["captures"] = caps
        self.save_meta()

    def capture_names(self) -> list[str]:
        return [c["name"] for c in self.meta["captures"]]

    def capture_info(self, name: str) -> dict:
        for c in self.meta["captures"]:
            if c["name"] == name:
                return c
        raise KeyError(f"capture '{name}' not in {self.path}")

    def load_capture(self, name: str) -> np.ndarray:
        x, _ = sf.read(self.path / self.capture_info(name)["file"], dtype="float64", always_2d=True)
        return x

    def load_stimulus(self, name: str) -> np.ndarray:
        x, _ = sf.read(self.path / self.meta["stimuli"][name]["file"], dtype="float64")
        return x

    def stim_params(self, name: str) -> dict:
        return self.meta["stimuli"][name]


def list_sessions(kind: str | None = None, root: Path = SESSIONS_DIR) -> list[Path]:
    paths = sorted(p for p in Path(root).glob("*") if (p / "meta.json").is_file())
    if kind:
        paths = [p for p in paths if json.loads((p / "meta.json").read_text())["kind"] == kind]
    return paths


def latest_calibration(root: Path = SESSIONS_DIR) -> Path:
    for p in reversed(list_sessions("calibration", root)):
        if json.loads((p / "meta.json").read_text()).get("complete"):
            return p
    raise FileNotFoundError("run: uv run pedalrig calibrate")
