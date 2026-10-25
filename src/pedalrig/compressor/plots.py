"""Compressor figures. Every function returns a matplotlib Figure and saves nothing."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _label(r) -> str:
    return r.get("label", r["name"])


def plot_static(results: list[dict], title="Compression curve (settled levels)") -> Figure:
    """Left: output vs input (dotted: uncompressed, i.e. low-level gain extended; dashed: threshold).
    Middle: gain reduction. Right: THD of the settled steps."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, r in enumerate(results):
        c = f"C{i}"
        a1.plot(r["in_dbu"], r["out_dbu"], "o-", c=c, ms=3, label=_label(r))
        a1.plot(r["in_dbu"], r["in_dbu"] + r["low_gain_db"], c=c, ls=":", lw=0.8)
        a2.plot(r["in_dbu"], r["gr_db"], "o-", c=c, ms=3, label=_label(r))
        if r["threshold_dbu"] is not None:
            for a in (a1, a2):
                a.axvline(r["threshold_dbu"], c=c, ls="--", lw=0.8)
        a3.plot(r["in_dbu"], r["thd_db"], "o-", c=c, ms=3, label=_label(r))
    a1.set_ylabel("output dBu")
    a1.set_title(title + "\ndotted: no compression, dashed: threshold (1 dB reduction)")
    a2.set_ylabel("gain reduction dB")
    a2.set_title("Gain reduction re low-level gain")
    a3.set_ylabel("THD dB re fundamental")
    a3.set_title("Distortion of the settled output")
    for a in (a1, a2, a3):
        a.set_xlabel("input dBu (send-referred)")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_steps(results: list[dict], title="Gain after a 20 dB level jump") -> Figure:
    """Gain trace of the first attack (left) and first release (right) of each result."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    for i, r in enumerate(results):
        for a, kind in ((a1, "attack"), (a2, "release")):
            tr = next((d for d in r["transitions"] if d["kind"] == kind), None)
            if tr is None:
                continue
            a.plot(tr["t_rel"] * 1e3, tr["gain_db"], c=f"C{i}", lw=0.9, label=_label(r))
            for k, ls in (("t63_s", "--"), ("t90_s", ":")):
                if tr[k] is not None:
                    a.axvline(tr[k] * 1e3, c=f"C{i}", ls=ls, lw=0.7)
    a1.set_xlim(-20, 300)
    a1.set_title(f"{title}: attack (up)\ndashed: 63 %, dotted: 90 % of the change")
    a2.set_xscale("symlog", linthresh=10)
    a2.set_title(f"{title}: release (down)\ndashed: 63 %, dotted: 90 % of the change")
    for a in (a1, a2):
        a.set_xlabel("ms after the jump")
        a.set_ylabel("gain dB (output − input)")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_hold(res: dict) -> Figure:
    """Release after a short and a long loud note (program-dependent release)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, d in enumerate(t for t in res["transitions"] if t["kind"] == "release"):
        ax.plot(d["t_rel"] * 1e3, d["gain_db"] - d["g_end_db"], c=f"C{i}",
                label=f"after {d['hold_s'] * 1e3:.0f} ms loud: t63 "
                      + (f"{d['t63_s'] * 1e3:.0f} ms" if d["t63_s"] is not None else "n/a"))
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("ms after the level drops")
    ax.set_ylabel("gain re final dB")
    ax.set_title("Release after a short vs a long loud note")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return fig


def plot_decay(results: list[dict], title="Plucked-note decay (200 Hz, 10 dB/s in)") -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, r in enumerate(results):
        ax.plot(r["t"], r["out_dbu"], c=f"C{i}", lw=0.9, label=f"{_label(r)} output")
    r0 = results[0]
    ax.plot(r0["t"], r0["in_dbu"], "k--", lw=0.8, label="input")
    ax.set_xlabel("s")
    ax.set_ylabel("per-cycle level dBu")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return fig


def plot_noise(results: list[dict], title="Output noise, no input") -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, r in enumerate(results):
        ax.semilogx(r["psd_f"], r["psd_dbfs"], c=f"C{i}", lw=0.8,
                    label=f"{_label(r)}: {r['noise_dbu']:.1f} dBu")
    r0 = results[0]
    ax.semilogx(r0["psd_f"], r0["floor_psd_dbfs"], "k--", lw=0.7, label=f"loopback floor: {r0['floor_dbu']:.1f} dBu")
    ax.set_xlim(20, 20000)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dBFS per bin")
    ax.set_title(title + " (RMS 20 Hz–20 kHz in legend)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    return fig


def plot_gr_vs_freq(curves: list[tuple[str, np.ndarray, np.ndarray]],
                    title="Gain reduction vs frequency") -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    for lab, f, gr in curves:
        ax.semilogx(f, gr, label=lab)
    ax.set_xlim(20, 20000)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB (small-signal gain − loud gain)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return fig
