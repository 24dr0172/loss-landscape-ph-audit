#!/usr/bin/env python3
"""Figure 1 motivating helix failure-mode audit.

Purpose
-------
Create the three-panel motivating figure showing why visual loops and ambient
Euclidean persistent homology can produce a false H1 signal on a contractible
helix, while graph-geodesic persistent homology suppresses the shortcut.

Decision convention
-------------------
Formal triggers are defined by pperm < ALPHA. The separation flag, TSR, zrob,
delta, and null-collapse diagnostics are descriptive only.

Outputs
-------
Writes fig1_motivation.pdf/png and fig1_motivation_stats.csv under figures/.
"""

from __future__ import annotations
# Shared audit primitives.
from pathlib import Path as _AuditPath
import sys as _audit_sys
_audit_sys.path.insert(0, str(_AuditPath(__file__).resolve().parents[1]))
from audit_common import (
    compute_stats,
    matched_step_nulls as null_matched_steps,
    geodesic_distance_matrix,
    safe_lifetime,
)

from pathlib import Path
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from joblib import Parallel, delayed
from ripser import ripser
from scipy.spatial.distance import pdist


# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig1_motivation.pdf"
OUT_PNG = OUT_DIR / "fig1_motivation.png"
OUT_CSV = OUT_DIR / "fig1_motivation_stats.csv"

RNG_SEED = 7

# Helix parameters
T_FULL = 1200
N_SUB = 400
TURNS = 2.5
RADIUS = 1.0
PITCH = 1.5
SIGMA = 0.03

# PH / null parameters
N_NULLS = 300
ALPHA = 0.05
EUCLIDEAN_PCT = 95.0
GEODESIC_K = 12
GEODESIC_MAX_K = 50
GEODESIC_PCT = 95.0

# Descriptive diagnostics only
DELTA_MIN = 1e-3
LOBS_MULT = 5.0
EPS = 1e-12
EPS_SIG = 1e-6
MAD_SCALE = 1.4826

# Parallelism
N_JOBS = -1


# =============================================================================
# BASIC HELPERS
# =============================================================================

def top_h1_bar(dgms, threshold: float) -> np.ndarray | None:
    """Largest H1 bar, using the filtration cutoff for censored deaths."""
    if len(dgms) < 2 or len(dgms[1]) == 0:
        return None
    H1 = np.asarray(dgms[1], dtype=float)
    deaths = np.where(np.isfinite(H1[:, 1]), H1[:, 1], float(threshold))
    lifetimes = np.maximum(deaths - H1[:, 0], 0.0)
    if len(lifetimes) == 0:
        return None
    i = int(np.argmax(lifetimes))
    return np.array([H1[i, 0], deaths[i]], dtype=float)


def adaptive_euclidean_thresh(X: np.ndarray, pct: float = 95.0) -> float | None:
    if len(X) < 2:
        return None
    d = pdist(X)
    if len(d) == 0:
        return None
    return float(np.percentile(d, pct))


# =============================================================================
# DATA AND NULL MODEL
# =============================================================================

def generate_helix(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a noisy 3D helix and return:
        X_full: full T_FULL-point trajectory
        X_sub : stride-subsampled N_SUB-point trajectory used for PH/plotting
    """
    t_full = np.linspace(0, TURNS * 2 * np.pi, T_FULL)
    X_full = np.column_stack([
        RADIUS * np.cos(t_full),
        RADIUS * np.sin(t_full),
        (PITCH / (2 * np.pi)) * t_full,
    ])
    X_full += rng.normal(0.0, SIGMA, X_full.shape)

    idx = np.linspace(0, T_FULL - 1, N_SUB, dtype=int)
    X_sub = X_full[idx]

    return X_full, X_sub




# =============================================================================
# PH COMPUTATION
# =============================================================================

def euclidean_ph(X: np.ndarray):
    """Ambient Euclidean PH and the finite filtration threshold used."""
    thresh = adaptive_euclidean_thresh(X, EUCLIDEAN_PCT)
    if thresh is None:
        thresh = float("inf")
        dgms = ripser(X, maxdim=1)["dgms"]
    else:
        dgms = ripser(X, maxdim=1, thresh=thresh)["dgms"]
    return dgms, thresh


def max_h1_euclidean(X: np.ndarray) -> float:
    dgms, thresh = euclidean_ph(X)
    return safe_lifetime(dgms, censoring_threshold=thresh)




def geodesic_ph(X: np.ndarray):
    """
    Graph-geodesic PH with median rescaling and percentile threshold.
    """
    D, final_k, sentinel_used = geodesic_distance_matrix(
        X,
        k0=GEODESIC_K,
        kmax=GEODESIC_MAX_K,
        sentinel_fill=False,
        rescale=False,
        eps=EPS,
    )

    positive = D[D > 0]
    if len(positive) == 0:
        return [np.empty((0, 2)), np.empty((0, 2))], D, final_k, sentinel_used

    med = float(np.median(positive))
    D_scaled = D / (med + EPS)

    offdiag_positive = D_scaled[D_scaled > EPS]
    thresh = (
        float(np.percentile(offdiag_positive, GEODESIC_PCT))
        if len(offdiag_positive) > 0
        else 1.0
    )

    dgms = ripser(
        D_scaled,
        maxdim=1,
        distance_matrix=True,
        thresh=thresh,
    )["dgms"]

    return dgms, D_scaled, final_k, sentinel_used, thresh


def max_h1_geodesic(X: np.ndarray) -> float:
    dgms, _, _, _, thresh = geodesic_ph(X)
    return safe_lifetime(dgms, censoring_threshold=thresh)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # -------------------------------------------------------------------------
    # Generate observed trajectory and nulls
    # -------------------------------------------------------------------------
    _, helix3 = generate_helix(rng)
    helix2 = helix3[:, :2]

    null_rng = np.random.default_rng(RNG_SEED + 1000)
    nulls = null_matched_steps(helix3, N_NULLS, null_rng)

    # -------------------------------------------------------------------------
    # Observed PH
    # -------------------------------------------------------------------------
    dgms_euc, thresh_euc = euclidean_ph(helix3)
    dgm0_euc = dgms_euc[0]
    dgm1_euc = dgms_euc[1] if len(dgms_euc) > 1 else np.empty((0, 2))

    dgms_geo, _, final_k, sentinel_used, thresh_geo = geodesic_ph(helix3)
    dgm0_geo = dgms_geo[0]
    dgm1_geo = dgms_geo[1] if len(dgms_geo) > 1 else np.empty((0, 2))

    Lobs_euc = safe_lifetime(dgms_euc, censoring_threshold=thresh_euc)
    Lobs_geo = safe_lifetime(dgms_geo, censoring_threshold=thresh_geo)

    top_h1_euc = top_h1_bar(dgms_euc, thresh_euc)

    # Fallback visual bar only if something very unexpected happens.
    if top_h1_euc is None:
        top_h1_euc = np.array([0.0, max(Lobs_euc, 1e-3)])

    # -------------------------------------------------------------------------
    # Null PH
    # -------------------------------------------------------------------------
    print("=" * 72)
    print("Figure 1 computation")
    print("=" * 72)
    print(f"N_SUB={N_SUB}, N_NULLS={N_NULLS}, GEODESIC_K={GEODESIC_K}")
    print("Computing Euclidean null lifetimes...")

    Lnull_euc = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(max_h1_euclidean)(Xn) for Xn in nulls
        ),
        dtype=float,
    )

    print("Computing graph-geodesic null lifetimes...")
    Lnull_geo = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(max_h1_geodesic)(Xn) for Xn in nulls
        ),
        dtype=float,
    )

    stats_euc = compute_stats(Lobs_euc, Lnull_euc)
    stats_geo = compute_stats(Lobs_geo, Lnull_geo)

    print()
    print("[Ambient Euclidean on 3D helix]")
    print(
        f"  Lobs={stats_euc['Lobs']:.6f}  "
        f"null_med={stats_euc['null_med']:.6f}  "
        f"null_max={stats_euc['null_max']:.6f}  "
        f"pperm={stats_euc['pperm']:.4f}  "
        f"formal={stats_euc['formal_trigger']}"
    )

    print("[Graph-geodesic on 3D helix]")
    print(
        f"  Lobs={stats_geo['Lobs']:.6f}  "
        f"null_med={stats_geo['null_med']:.6f}  "
        f"null_max={stats_geo['null_max']:.6f}  "
        f"pperm={stats_geo['pperm']:.4f}  "
        f"formal={stats_geo['formal_trigger']}  "
        f"final_k={final_k}  sentinel_used={sentinel_used}"
    )

    # Save figure stats for auditability.
    import pandas as pd
    pd.DataFrame([
        {"metric": "ambient_euclidean_3d", **stats_euc},
        {"metric": "graph_geodesic_3d", **stats_geo,
         "final_k": final_k, "sentinel_used": sentinel_used},
    ]).to_csv(OUT_CSV, index=False)

    # Manuscript-facing safety check: the motivating figure must exhibit the
    # intended controlled pathology. Save the audit CSV first, then fail loudly
    # if the expected qualitative result is not obtained.
    if not stats_euc["formal_trigger"]:
        raise RuntimeError(
            "Figure 1 validation failed: ambient Euclidean PH did not formally "
            "trigger on the helix metric-shortcut control. See " + str(OUT_CSV)
        )
    if stats_geo["formal_trigger"]:
        raise RuntimeError(
            "Figure 1 validation failed: graph-geodesic PH formally triggered "
            "on the contractible helix. See " + str(OUT_CSV)
        )

    # -------------------------------------------------------------------------
    # Plot styling
    # -------------------------------------------------------------------------
    C_TRAJ = "#2166AC"
    C_FP = "#D73027"
    C_OK = "#4DAC26"
    C_H0 = "#999999"
    C_BG = "#F6F6F6"

    fig = plt.figure(figsize=(14, 4.8))
    gs = GridSpec(
        1, 3, figure=fig, wspace=0.44,
        left=0.045, right=0.97, top=0.83, bottom=0.13,
    )
    ax_A, ax_B, ax_C = (fig.add_subplot(gs[i]) for i in range(3))

    # -------------------------------------------------------------------------
    # Panel (a): 2D projection
    # -------------------------------------------------------------------------
    ax_A.set_facecolor(C_BG)

    theta_ring = np.linspace(0, 2 * np.pi, 300)
    ax_A.fill(
        RADIUS * np.cos(theta_ring),
        RADIUS * np.sin(theta_ring),
        color=C_FP,
        alpha=0.06,
        zorder=1,
    )

    ax_A.scatter(
        helix2[:, 0],
        helix2[:, 1],
        c=np.arange(N_SUB),
        cmap="plasma_r",
        s=11,
        zorder=3,
        linewidths=0,
    )
    ax_A.plot(
        helix2[:, 0],
        helix2[:, 1],
        lw=0.8,
        color=C_TRAJ,
        alpha=0.35,
        zorder=2,
    )

    ax_A.scatter(
        *helix2[0],
        s=80,
        color="white",
        edgecolors=C_TRAJ,
        lw=2.0,
        zorder=6,
    )
    ax_A.scatter(
        *helix2[-1],
        s=80,
        color=C_TRAJ,
        edgecolors="white",
        lw=2.0,
        zorder=6,
    )

    ax_A.text(
        helix2[0, 0] - 0.09,
        helix2[0, 1] + 0.13,
        "start",
        fontsize=7.5,
        color=C_TRAJ,
        ha="center",
    )
    ax_A.text(
        helix2[-1, 0] + 0.09,
        helix2[-1, 1] - 0.17,
        "end",
        fontsize=7.5,
        color=C_TRAJ,
        ha="center",
    )

    ax_A.annotate(
        "",
        xy=(0.05, 0.88),
        xytext=(-0.05, -0.88),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.50",
            color=C_FP,
            lw=1.7,
        ),
    )

    ax_A.text(
        0.97,
        0.04,
        "Looks like a loop\n(projection artifact)",
        transform=ax_A.transAxes,
        fontsize=8.2,
        color=C_FP,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FP, alpha=0.92),
    )

    ax_A.set_aspect("equal")
    ax_A.set_xlabel("$x$ (drops $z$-drift)", fontsize=9)
    ax_A.set_ylabel("$y$ (drops $z$-drift)", fontsize=9)
    ax_A.set_title(
        "(a) Drop-$z$ projection of 3D helix\n"
        "(contractible path looks circular)",
        fontsize=10.5,
        fontweight="bold",
        pad=7,
    )
    ax_A.tick_params(labelsize=8)

    # -------------------------------------------------------------------------
    # Panel (b): Ambient 3D Euclidean PH
    # -------------------------------------------------------------------------
    ax_B.set_facecolor(C_BG)

    h0_fin = dgm0_euc[dgm0_euc[:, 1] < np.inf]
    if len(h0_fin) > 0:
        h0_life = h0_fin[:, 1] - h0_fin[:, 0]
        h0_show = h0_fin[np.argsort(h0_life)[-4:]]
    else:
        h0_show = np.empty((0, 2))

    bars_B = [(b, d, C_H0, "$H_0$") for b, d in sorted(h0_show, key=lambda r: r[0])]
    bars_B.append((top_h1_euc[0], top_h1_euc[1], C_FP, "$H_1$"))

    max_d_B = max(float(top_h1_euc[1]) * 1.10, 1e-3)

    for i, (b, d, col, lbl) in enumerate(bars_B):
        ax_B.barh(
            i,
            max(min(d, max_d_B) - b, 0.0),
            left=b,
            height=0.56,
            color=col,
            alpha=0.85 if col == C_FP else 0.45,
            edgecolor="none",
        )

    ax_B.set_xlim(-0.02 * max_d_B, max_d_B)
    ax_B.set_ylim(-0.65, len(bars_B) - 0.35)
    ax_B.set_yticks(range(len(bars_B)))
    ax_B.set_yticklabels([b[3] for b in bars_B], fontsize=10)
    ax_B.set_xlabel("Filtration radius (ambient Euclidean)", fontsize=9)
    ax_B.set_title(
        "(b) Ambient 3D Euclidean PH\n"
        "(metric-shortcut false positive)",
        fontsize=10.5,
        fontweight="bold",
        pad=7,
    )
    ax_B.tick_params(labelsize=8)

    h1_row = len(bars_B) - 1
    h1_mid = (top_h1_euc[0] + min(top_h1_euc[1], max_d_B)) / 2

    ax_B.annotate(
        f"Long-lived $H_1$\n"
        f"(lifetime $\\approx$ {stats_euc['Lobs']:.2f})\n"
        f"metric shortcut",
        xy=(h1_mid, h1_row),
        xytext=(max_d_B * 0.38, max(h1_row - 1.4, 0.1)),
        fontsize=8,
        color=C_FP,
        arrowprops=dict(arrowstyle="-|>", color=C_FP, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_FP, alpha=0.92),
    )

    euc_decision_text = (
        "formal trigger"
        if stats_euc["formal_trigger"]
        else "no formal trigger"
    )
    ax_B.text(
        0.97,
        0.04,
        f"$p_\\mathrm{{perm}} = {stats_euc['pperm']:.3f}$\n"
        f"({euc_decision_text})",
        transform=ax_B.transAxes,
        fontsize=8,
        color=C_FP,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FP, alpha=0.92),
    )

    lh_B = [
        mpatches.Patch(color=C_H0, alpha=0.55, label="$H_0$ (components)"),
        mpatches.Patch(color=C_FP, alpha=0.85, label="$H_1$ (spurious loop)"),
    ]
    ax_B.legend(
        handles=lh_B,
        fontsize=8.2,
        loc="lower left",
        framealpha=0.92,
        edgecolor="none",
    )

    # -------------------------------------------------------------------------
    # Panel (c): Graph-geodesic PH
    # -------------------------------------------------------------------------
    ax_C.set_facecolor(C_BG)

    geo_h0_fin = dgm0_geo[dgm0_geo[:, 1] < np.inf]
    if len(geo_h0_fin) > 0:
        geo_h0_life = geo_h0_fin[:, 1] - geo_h0_fin[:, 0]
        geo_h0_show = geo_h0_fin[np.argsort(geo_h0_life)[-4:]]
    else:
        geo_h0_show = np.array([[0.0, 1e-3]])

    bars_C = [(b, d, C_OK, "$H_0$") for b, d in sorted(geo_h0_show, key=lambda r: r[0])]
    max_d_C = max(max(d for _, d, *_ in bars_C) * 1.3, 1e-3)

    for i, (b, d, col, lbl) in enumerate(bars_C):
        ax_C.barh(
            i,
            max(min(d, max_d_C) - b, 0.0),
            left=b,
            height=0.56,
            color=col,
            alpha=0.55,
            edgecolor="none",
        )

    h1_c = len(bars_C)

    if len(dgm1_geo) > 0 and stats_geo["Lobs"] > 0:
        # Show up to three longest H1 bars. Classes alive at the finite
        # cutoff are plotted to the cutoff and treated as right-censored.
        geo_h1_plot = np.asarray(dgm1_geo, dtype=float).copy()
        geo_h1_plot[:, 1] = np.where(
            np.isfinite(geo_h1_plot[:, 1]),
            geo_h1_plot[:, 1],
            thresh_geo,
        )
        if len(geo_h1_plot) > 0:
            lives = geo_h1_plot[:, 1] - geo_h1_plot[:, 0]
            show_h1 = geo_h1_plot[np.argsort(lives)[-3:]]
            for b, d in show_h1:
                ax_C.barh(
                    h1_c,
                    max(min(d, max_d_C * 0.18) - b, 0.0),
                    left=b,
                    height=0.40,
                    color=C_FP,
                    alpha=0.30,
                    edgecolor="none",
                )

        ax_C.text(
            max_d_C * 0.08,
            h1_c,
            rf"$H_1$ at numerical scale"
            "\n"
            rf"(max lifetime $\approx {stats_geo['Lobs']:.2e}$)",
            va="center",
            fontsize=8.7,
            color=C_OK,
            style="italic",
            fontweight="bold",
        )
    else:
        ax_C.text(
            0.03 * max_d_C,
            h1_c,
            r"$H_1 = \varnothing$ (no loop detected)",
            va="center",
            fontsize=9,
            color=C_OK,
            style="italic",
            fontweight="bold",
        )

    ax_C.set_xlim(-0.02 * max_d_C, max_d_C)
    ax_C.set_ylim(-0.65, h1_c + 0.5)
    ax_C.set_yticks(list(range(len(bars_C))) + [h1_c])
    ax_C.set_yticklabels([b[3] for b in bars_C] + ["$H_1$"], fontsize=10)
    ax_C.set_xlabel("Filtration radius (geodesic, median-rescaled)", fontsize=9)
    ax_C.set_title(
        "(c) Graph-geodesic PH\n"
        "(metric shortcut suppressed)",
        fontsize=10.5,
        fontweight="bold",
        pad=7,
    )
    ax_C.tick_params(labelsize=8)

    geo_decision_text = (
        "formal trigger"
        if stats_geo["formal_trigger"]
        else "no formal signal"
    )
    ax_C.text(
        0.97,
        0.04,
        f"$p_\\mathrm{{perm}} = {stats_geo['pperm']:.3f}$\n"
        f"({geo_decision_text})",
        transform=ax_C.transAxes,
        fontsize=8,
        color=C_OK,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_OK, alpha=0.92),
    )

    lh_C = [
        mpatches.Patch(color=C_OK, alpha=0.6, label="$H_0$ (components)"),
        mpatches.Patch(
            color="white",
            ec=C_OK,
            lw=1.5,
            label="$H_1$ suppressed by geodesic metric",
        ),
    ]
    ax_C.legend(
        handles=lh_C,
        fontsize=8.2,
        loc="lower right",
        framealpha=0.92,
        edgecolor="none",
    )

    # -------------------------------------------------------------------------
    # Suptitle and output
    # -------------------------------------------------------------------------
    fig.text(
        0.5,
        0.975,
        "Why visual loops and Euclidean PH are not reliable recurrence evidence",
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
    )

    fig.savefig(OUT_PDF, dpi=250, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")

    print()
    print(f"Written: {OUT_PDF}")
    print(f"Written: {OUT_PNG}")
    print(f"Stats  : {OUT_CSV}")


if __name__ == "__main__":
    main()
