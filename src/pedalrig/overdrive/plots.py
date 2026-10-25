"""Overdrive figures. Every function returns a matplotlib Figure and saves nothing."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_amp_transfer(results: list[dict], title="Amplitude transfer (per-cycle fundamental)") -> Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, r in enumerate(results):
        c = f"C{i}"
        lab = r["label"] if "label" in r else r["name"]
        a1.plot(r["in_dbu"], r["out_dbu"], c=c, label=lab)
        a1.plot(r["in_dbu"], r["in_dbu"] + r["linear_gain_db"], c=c, ls=":", lw=0.8)
        if r["onset_1db_dbu"] is not None:
            a1.axvline(r["onset_1db_dbu"], c=c, ls="--", lw=0.8)
        a2.plot(r["in_dbu"], r["thd_db"], c=c, label=f"{lab} THD")
        if r["hd2_db"] is not None:
            a2.plot(r["in_dbu"], r["hd2_db"], c=c, ls=":", lw=0.8)
        if r["hd3_db"] is not None:
            a2.plot(r["in_dbu"], r["hd3_db"], c=c, ls="-.", lw=0.8)
    a1.set_xlabel("input dBu (send-referred)")
    a1.set_ylabel("output dBu")
    a1.set_title(title + "\ndotted: linear extrapolation, dashed: 1 dB compression onset")
    a2.set_xlabel("input dBu")
    a2.set_ylabel("dB re fundamental")
    a2.set_title("THD (solid), HD2 (dotted), HD3 (dash-dot)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_asym(results: list[dict]) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for r in results:
        ax.plot(r["in_dbu"], r["asym_db"], label=r.get("label", r["name"]))
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("input dBu")
    ax.set_ylabel("20log10(peak+/|peak-|) dB")
    ax.set_title("Asymmetry (0 = symmetric clipping)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return fig


def plot_scatter_transfer(results: list[dict], max_points: int = 200_000) -> Figure:
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.2), squeeze=False)
    for ax, r in zip(axes[0], results):
        x, y = r["x"], r["y"]
        step = max(1, len(x) // max_points)
        ax.plot(x[::step], y[::step], ",", alpha=0.05, rasterized=True)
        lim = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("input (FS units at send)")
        ax.set_ylabel("output (FS units)")
        ax.set_title(r.get("label", r["name"]) + (" (inverted)" if r["inverted"] else ""), fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bursts(res: dict) -> Figure:
    cap, ref = res["cap"], res["ref"]
    t = cap["t_ms"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for i, e in enumerate(cap["env_dbu"]):
        a1.plot(t, e, c="C0", alpha=0.4 + 0.6 * i / len(cap["env_dbu"]), lw=0.9)
    a1.plot(ref["t_ms"], ref["env_dbu"].mean(axis=0), "k--", lw=0.8, label="loopback ref (mean)")
    a1.set_xlabel("ms into burst")
    a1.set_ylabel("per-cycle level dBu")
    a1.set_title(f"Burst envelopes (8 bursts, darker = later)\nsag per burst: "
                 + ", ".join(f"{s:+.2f}" for s in cap["sag_db"]) + " dB")
    a1.legend(fontsize=7)
    for i, m in enumerate(cap["mean_rel"]):
        a2.plot(t, m, c="C1", alpha=0.4 + 0.6 * i / len(cap["mean_rel"]), lw=0.9)
    a2.set_xlabel("ms into burst")
    a2.set_ylabel("per-cycle mean / peak+")
    a2.set_title("Bias shift (0 = no DC shift)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig
