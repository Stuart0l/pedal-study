"""Figures for the notebooks. Every function returns a matplotlib Figure and saves nothing."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .config import SEND_DBU_AT_0DBFS


def _logx(ax, lo=20, hi=20000):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Hz")


def plot_psd(f, psd_dbfs, hum: dict | None = None, title="Noise spectrum") -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(f[1:], psd_dbfs[1:], lw=0.8)
    for hz, v in (hum or {}).items():
        ax.plot(hz, v, "o", ms=4, label=f"{hz} Hz {v:.0f} dBFS")
    _logx(ax, 5, f[-1])
    ax.set_ylabel("dBFS (sine-peak ref, 1 Hz bins)")
    ax.set_title(title)
    if hum:
        ax.legend(fontsize=7, ncol=3)
    return fig


def plot_levels(rows: list[dict], G_db: float | None = None) -> Figure:
    lv = [r["in_dbfs"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(lv, [r["gain_db"] for r in rows], "o-")
    if G_db is not None:
        a1.axhline(G_db, ls="--", c="k", lw=0.8, label=f"G = {G_db:.3f} dB")
        a1.legend()
    a1.set_xlabel("stimulus dBFS")
    a1.set_ylabel("loop gain dB")
    a1.set_title("Gain vs level (1 kHz)")
    a2.plot(lv, [r["thdn_db"] for r in rows], "o-", label="THD+N")
    for k in (2, 3):
        a2.plot(lv, [r[f"hd{k}_db"] for r in rows], ".--", label=f"HD{k}")
    a2.set_xlabel("stimulus dBFS")
    a2.set_ylabel("dB re fundamental")
    a2.set_title("Distortion vs level (1 kHz)")
    a2.legend()
    for a in (a1, a2):
        a.grid(alpha=0.3)
        sec = a.secondary_xaxis("top", functions=(lambda v: v + SEND_DBU_AT_0DBFS, lambda v: v - SEND_DBU_AT_0DBFS))
        sec.set_xlabel("send-referred dBu (per spec)")
    fig.tight_layout()
    return fig


def plot_fr(curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]], title="Frequency response",
            lo=20, hi=20000, ylim=None) -> Figure:
    """curves: (label, f, mag_db, phase_deg or None)."""
    has_phase = any(c[3] is not None for c in curves)
    fig, axes = plt.subplots(2 if has_phase else 1, 1, figsize=(9, 6 if has_phase else 4), sharex=True)
    axes = np.atleast_1d(axes)
    ph_band = []
    for label, f, mag, ph in curves:
        axes[0].plot(f, mag, label=label)
        if has_phase and ph is not None:
            axes[1].plot(f, ph, label=label)
            ph_band.append(ph[(f >= lo) & (f <= min(hi, 20000))])
    axes[0].set_ylabel("dB")
    axes[0].set_title(title)
    if ylim:
        axes[0].set_ylim(*ylim)
    axes[0].legend(fontsize=8)
    if has_phase:
        axes[1].set_ylabel("phase (deg; 2π-wrapped at 1 kHz, −180 = inverted)")
        allp = np.concatenate(ph_band)  # above 20 kHz the anti-alias filter's phase swamps the scale
        pad = 0.05 * max(np.ptp(allp), 10)
        axes[1].set_ylim(allp.min() - pad, allp.max() + pad)
    for a in axes:
        _logx(a, lo, hi)
    fig.tight_layout()
    return fig


def plot_hd(curves: list[tuple[str, dict]], floor: dict | None = None, title="Harmonic distortion vs frequency",
            orders=(2, 3)) -> Figure:
    """curves: (label, hd_vs_freq result). Plots THD (solid) and HD orders (dotted)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (label, hd) in enumerate(curves):
        c = f"C{i}"
        ax.plot(hd["f"], hd["THD"], c=c, label=f"{label} THD")
        for n, ls in zip(orders, (":", "-.")):
            ax.plot(hd["f"], hd["HD"][n], c=c, ls=ls, lw=0.8, label=f"{label} HD{n}")
    if floor is not None:
        ax.plot(floor["f"], floor["THD"], "k--", lw=0.8, label="loopback THD (floor)")
    _logx(ax)
    ax.set_ylabel("dB re fundamental")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    return fig


def plot_repeat(grid, deltas: dict) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for n, d in deltas.items():
        ax.plot(grid, d, label=n)
    _logx(ax)
    ax.set_ylabel("dB re ref_sweep_m20")
    ax.set_title("Repeatability (1/12-octave smoothed)")
    ax.legend(fontsize=8)
    return fig
