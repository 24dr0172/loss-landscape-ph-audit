#!/usr/bin/env python3
"""Step-autocorrelation diagnostic.

Purpose
-------
Compute step-magnitude autocorrelation and lagged directional cosine similarity
for saved neural optimization trajectories. This diagnoses temporal dependence
in increments and motivates block-surrogate diagnostics.

Interpretation
--------------
This analysis is diagnostic only. It does not establish recurrence and does not
provide a formal recurrence null model.

Execution
---------
Use --stages to choose the architecture stages to process. The combined outputs
used by the manuscript are written under acf_results/combined/.
"""
# Shared audit primitives.
from pathlib import Path as _AuditPath
import sys as _audit_sys
_audit_sys.path.insert(0, str(_AuditPath(__file__).resolve().parents[1]))
from audit_common import (
    compute_stats,
    matched_step_null,
    matched_step_nulls,
    matched_step_nulls as null_matched_steps,
    block_null,
    block_step_nulls,
    block_permutation_indices,
    geodesic_distance_matrix,
    geodesic_h1_lifetime,
    h1_lifetime_from_distance_matrix,
    safe_lifetime,
    sha256_file,
    stride_subsample as audit_stride_subsample,
)
import argparse
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TRAJ_DIR   = Path("exp4_results/trajectories")
OUT_DIR    = Path("acf_results")
SEEDS      = [20042, 21042, 22042]
MAX_LAG    = 40
CHUNK_COLS = 50_000       # column-chunk width for ResNet direction ACF
ARCH_REGISTRY = {
    "mlp2": {
        "name":    "mlp2_mnist",
        "label":   "MLP2 / MNIST",
        "color":   "#1f77b4",
        "stage":   1,
        "subdir":  "stage1_mlp2",
        "chunked": False,       # D ~203 K: load steps directly
    },
    "smallcnn": {
        "name":    "smallcnn_cifar10",
        "label":   "SmallCNN / CIFAR-10",
        "color":   "#ff7f0e",
        "stage":   2,
        "subdir":  "stage2_smallcnn",
        "chunked": False,       # D ~321 K: load steps directly
    },
    "resnet18": {
        "name":    "resnet18_cifar10",
        "label":   "ResNet-18 / CIFAR-10",
        "color":   "#2ca02c",
        "stage":   3,
        "subdir":  "stage3_resnet18",
        "chunked": True,        # D ~11 M: must chunk direction ACF
    },
}
OPTIMIZERS = ["sgd", "sgd_momentum", "adam", "adamw"]
OPT_LABELS = {
    "sgd":          "SGD",
    "sgd_momentum": "SGD + momentum",
    "adam":         "Adam",
    "adamw":        "AdamW",
}
OPT_COLORS = {
    "sgd":          "#333333",
    "sgd_momentum": "#e377c2",
    "adam":         "#17becf",
    "adamw":        "#bcbd22",
}
OPT_LS = {          # dashed for plain SGD to distinguish it visually
    "sgd":          "--",
    "sgd_momentum": "-",
    "adam":         "-",
    "adamw":        "-",
}
# Next-stage suggestion threshold: proceed to next architecture stage
# if at least one optimizer shows |C[1:20]| > this value.
# This is a workflow heuristic only, not a statistical test.
NEXT_STAGE_COS_THRESHOLD = 0.10
# ---------------------------------------------------------------------------
# ACF CORE
# ---------------------------------------------------------------------------
def _acf_1d(x, max_lag):
    """
    Unbiased FFT-based ACF, lags 0..max_lag. Lag-0 always 1.0.
    Returns NaN array (with lag-0=1.0) for constant or too-short series.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        out = np.full(max_lag + 1, np.nan)
        out[0] = 1.0
        return out
    x = x - x.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    F   = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(F * np.conj(F))[:max_lag + 1].copy()
    acf /= np.array([n - k for k in range(max_lag + 1)], dtype=np.float64)
    if abs(acf[0]) < 1e-12:          # constant series
        out = np.full(max_lag + 1, np.nan)
        out[0] = 1.0
        return out
    acf /= acf[0]
    return acf
def magnitude_acf(X, max_lag):
    """
    ACF of ||Delta_t|| = ||w_{t+1} - w_t||.
    X: (T, D) float32 memmap. Always chunked to avoid float64 copy of full X.
    Returns lags 0..max_lag, shape (max_lag+1,).
    """
    T, D   = X.shape
    mag_sq = np.zeros(T - 1, dtype=np.float64)
    for cs in range(0, D, CHUNK_COLS):
        ce   = min(cs + CHUNK_COLS, D)
        diff = np.diff(X[:, cs:ce].astype(np.float64), axis=0)
        mag_sq += np.einsum('ij,ij->i', diff, diff)
    return _acf_1d(np.sqrt(mag_sq), max_lag)
def direction_acf_direct(X, max_lag):
    """
    Mean cosine similarity of step directions at lags 1..max_lag.
    For MLP2/SmallCNN: loads full (T-1, D) step matrix at float64.
    Returns shape (max_lag,) indexed lag 1, 2, ..., max_lag.
    """
    steps   = np.diff(X.astype(np.float64), axis=0)   # (T-1, D)
    mags    = np.linalg.norm(steps, axis=1) + 1e-12
    T_steps = len(steps)
    cos_out = np.zeros(max_lag, dtype=np.float64)
    for tau in range(1, max_lag + 1):
        n = T_steps - tau
        if n <= 0:
            cos_out[tau - 1] = np.nan
            continue
        dots = np.einsum('ij,ij->i', steps[:n], steps[tau:])
        cos_out[tau - 1] = float(np.mean(dots / (mags[:n] * mags[tau:])))
    return cos_out
def direction_acf_chunked(X, max_lag):
    """
    Chunked direction ACF for ResNet-18 (D ~11 M).
    Peak RAM = 2 * T_steps * chunk_cols * 8 bytes + max_lag * T_steps * 8.
    Returns shape (max_lag,) indexed lag 1, 2, ..., max_lag.
    """
    T, D    = X.shape
    T_steps = T - 1
    mag_sq    = np.zeros(T_steps, dtype=np.float64)
    dot_accum = np.zeros((max_lag, T_steps), dtype=np.float64)
    for cs in range(0, D, CHUNK_COLS):
        ce     = min(cs + CHUNK_COLS, D)
        steps_c = np.diff(X[:, cs:ce].astype(np.float64), axis=0)
        mag_sq += np.einsum('ij,ij->i', steps_c, steps_c)
        for tau in range(1, max_lag + 1):
            n = T_steps - tau
            if n <= 0:
                continue
            dot_accum[tau - 1, :n] += np.einsum(
                'ij,ij->i', steps_c[:n], steps_c[tau:]
            )
    mags    = np.sqrt(mag_sq)
    cos_out = np.zeros(max_lag, dtype=np.float64)
    for tau in range(1, max_lag + 1):
        n = T_steps - tau
        if n <= 0:
            cos_out[tau - 1] = np.nan
            continue
        denom = mags[:n] * mags[tau:] + 1e-12
        cos_out[tau - 1] = float(
            np.mean(dot_accum[tau - 1, :n] / denom)
        )
    return cos_out
# ---------------------------------------------------------------------------
# SEED AUTO-DISCOVERY
# ---------------------------------------------------------------------------
def discover_seeds(traj_dir, arch_name):
    """
    Scan traj_dir for files matching {arch_name}__{opt}__seed{N}.npy
    and return a dict mapping optimizer -> sorted list of seed ints.
    Ignores __metrics.npy files.
    Example files:
      mlp2_mnist__sgd__seed42.npy       -> {"sgd": [42, ...], ...}
      mlp2_mnist__adamw__seed342.npy    -> {"adamw": [342, ...], ...}
    Each optimizer may have a different set of seeds — this is common
    when training runs were seeded independently per optimizer.
    Returns: dict {opt: [seed, ...]} for opts present on disk.
    """
    import re
    traj_dir = Path(traj_dir)
    if not traj_dir.exists():
        return {}
    # Match: {arch}__{opt}__seed{N}.npy  (not ending in __metrics.npy)
    pattern = re.compile(
        rf"^{re.escape(arch_name)}__([^_](?:[^_]|_(?!seed))*?)__seed(\d+)\.npy$"
    )
    seeds_by_opt = {}
    for f in traj_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            opt  = m.group(1)
            seed = int(m.group(2))
            seeds_by_opt.setdefault(opt, set()).add(seed)
    return {opt: sorted(s) for opt, s in seeds_by_opt.items()}
# ---------------------------------------------------------------------------
# SINGLE-ARCHITECTURE ANALYSIS
# ---------------------------------------------------------------------------
def analyse_arch(arch_key, traj_dir, out_dir, seeds, max_lag):
    """
    Full ACF pipeline for one architecture.
    Writes CSVs + figures to out_dir/subdir.
    Returns DataFrame with columns:
        arch, optimizer, seed, lag, acf_mag, cos_sim
    If seeds is None or empty, seeds are auto-discovered from filenames.
    """
    cfg       = ARCH_REGISTRY[arch_key]
    arch_name = cfg["name"]
    label     = cfg["label"]
    stage_dir = out_dir / cfg["subdir"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    # -- Auto-discover seeds per optimizer if not provided -----------------
    if not seeds:
        seeds_by_opt = discover_seeds(traj_dir, arch_name)
        if not seeds_by_opt:
            print(f"  No trajectory files found in {traj_dir}.")
            print(f"  Check that NOTEBOOK_TRAJ_DIR points to the correct folder.")
            print(f"  Expected pattern: {arch_name}__<opt>__seed<N>.npy")
            return pd.DataFrame(
                columns=["arch", "optimizer", "seed", "lag", "acf_mag", "cos_sim"]
            )
        print(f"  Auto-discovered seeds for {arch_name}:")
        for opt, s in seeds_by_opt.items():
            print(f"    {opt:20s} -> {s}")
    else:
        # User supplied a flat list: use same seeds for all optimizers
        seeds_by_opt = {opt: list(seeds) for opt in OPTIMIZERS}
    # Total files to attempt
    n_total = sum(len(v) for v in seeds_by_opt.values())
    print("\n" + "=" * 70)
    print(f"  STAGE {cfg['stage']}  --  {label}")
    print(f"  Output directory: {stage_dir}")
    print(f"  Files to process : {n_total}")
    print("=" * 70)
    records = []
    done    = 0
    for opt in OPTIMIZERS:
        opt_seeds = seeds_by_opt.get(opt, [])
        if not opt_seeds:
            print(f"  [--] {opt}: no files found on disk, skipping.")
            continue
        for seed in opt_seeds:
            done += 1
            fname = traj_dir / f"{arch_name}__{opt}__seed{seed}.npy"
            if not fname.exists():
                print(f"  [{done}/{n_total}] SKIP (not found): {fname.name}")
                continue
            t0 = time.time()
            print(f"  [{done}/{n_total}] {opt} / seed {seed} ...",
                  end=" ", flush=True)
            X    = np.load(fname, mmap_mode='r')   # never fully materialised
            T, D = X.shape
            acf_mag  = magnitude_acf(X, max_lag)
            if cfg["chunked"]:
                cos_sims = direction_acf_chunked(X, max_lag)
            else:
                cos_sims = direction_acf_direct(X, max_lag)
            elapsed = time.time() - t0
            print(
                f"D={D:,}  T={T}  "
                f"acf_mag[lag1]={acf_mag[1]:+.3f}  "
                f"cos[lag1]={cos_sims[0]:+.3f}  "
                f"({elapsed:.1f}s)"
            )
            for lag in range(0, max_lag + 1):
                records.append({
                    "arch":      arch_name,
                    "optimizer": opt,
                    "seed":      seed,
                    "lag":       lag,
                    "acf_mag":   float(acf_mag[lag]),
                    # cos_sims is indexed 0..max_lag-1 (= lags 1..max_lag)
                    "cos_sim":   float(cos_sims[lag - 1]) if lag >= 1 else 1.0,
                })
    if not records:
        print(f"  No trajectory files found for {label}.")
        return pd.DataFrame(
            columns=["arch", "optimizer", "seed", "lag", "acf_mag", "cos_sim"]
        )
    df = pd.DataFrame(records)
    # -- CSVs ---------------------------------------------------------------
    df.to_csv(stage_dir / "acf_summary.csv", index=False)
    stats = (
        df.groupby(["arch", "optimizer", "lag"])[["acf_mag", "cos_sim"]]
          .agg(["mean", "std"])
          .round(4)
          .reset_index()
    )
    stats.to_csv(stage_dir / "acf_summary_stats.csv", index=False)
    print(f"\n  CSVs saved to {stage_dir}")
    # -- Figures ------------------------------------------------------------
    _plot_block_overlap(df, stage_dir / "acf_block_overlap.pdf",
                        max_lag, label)
    _plot_per_optimizer(df, "acf_mag",
                        ylabel="ACF of step magnitudes ||Delta_t||",
                        title=f"{label} -- Step Magnitude ACF",
                        out_path=stage_dir / "acf_magnitude.pdf",
                        max_lag=max_lag)
    _plot_per_optimizer(df, "cos_sim",
                        ylabel="Mean cos(Delta_t, Delta_{{t+tau}})",
                        title=f"{label} -- Step Direction Cosine Similarity",
                        out_path=stage_dir / "acf_direction.pdf",
                        max_lag=max_lag)
    # -- Dependence report + window summary --------------------------------
    win_df      = _window_summary(df)
    win_df.to_csv(stage_dir / "acf_window_summary.csv", index=False)
    report_text = _dependence_report(df, label)
    (stage_dir / "dependence_report.txt").write_text(report_text)
    print(report_text)
    return df
# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def _plot_block_overlap(df, out_path, max_lag, arch_label):
    """
    PRIMARY FIGURE for Sec 6.7.
    Two panels: magnitude ACF (left) and direction cosine similarity (right).
    One line per optimizer. Gold band at lag 10, coral at lag 20.
    Mean across seeds shown; shaded band = +/- 1 SD.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    panels = [
        (axes[0], "acf_mag",  "ACF of step magnitudes ||Delta_t||"),
        (axes[1], "cos_sim",  "Mean cos(Delta_t, Delta_{t+tau})"),
    ]
    for ax, metric, ylabel in panels:
        mean_df = (df.groupby(["optimizer", "lag"])[metric]
                     .mean().reset_index())
        std_df  = (df.groupby(["optimizer", "lag"])[metric]
                     .std().fillna(0).reset_index()
                     .rename(columns={metric: "std"}))
        # Block-flip lag bands
        ax.axvspan( 9.5, 10.5, color="gold",  alpha=0.30,
                    zorder=0, label="b=10 block")
        ax.axvspan(19.5, 20.5, color="coral", alpha=0.25,
                    zorder=0, label="b=20 block")
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", zorder=1)
        for opt in OPTIMIZERS:
            sub_m = mean_df[mean_df["optimizer"] == opt].sort_values("lag")
            sub_s = std_df[ std_df["optimizer"]  == opt].sort_values("lag")
            if sub_m.empty:
                continue
            x  = sub_m["lag"].values
            y  = sub_m[metric].values
            sd = sub_s["std"].values if len(sub_s) == len(sub_m) \
                 else np.zeros_like(y)
            ax.plot(x, y,
                    color=OPT_COLORS[opt],
                    linewidth=2.0,
                    linestyle=OPT_LS[opt],
                    label=OPT_LABELS[opt],
                    zorder=2)
            ax.fill_between(x, y - sd, y + sd,
                            color=OPT_COLORS[opt], alpha=0.12, zorder=1)
        ax.set_xlabel("Lag tau (epochs)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0, max_lag)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, loc="upper right")
        # Annotate band labels
        ymax = ax.get_ylim()[1]
        ax.text(10, ymax * 0.97, "b=10", fontsize=7,
                color="goldenrod", ha="center", va="top")
        ax.text(20, ymax * 0.97, "b=20", fontsize=7,
                color="tomato", ha="center", va="top")
    fig.suptitle(
        f"{arch_label} -- Step ACF vs block-diagnostic lags (b in {{10, 20}})\n"
        "Gold/coral bands mark the block sizes used for dependence-preserving diagnostics. "
        "Mean +/- 1 SD across seeds.",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Figure saved: {out_path.name}")
def _plot_per_optimizer(df, metric, ylabel, title, out_path, max_lag):
    """2x2 grid: one panel per optimizer, one line per arch in df."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                             sharey=True, sharex=True)
    axes = axes.flatten()
    mean_df = (df.groupby(["arch", "optimizer", "lag"])[metric]
                 .mean().reset_index())
    std_df  = (df.groupby(["arch", "optimizer", "lag"])[metric]
                 .std().fillna(0).reset_index()
                 .rename(columns={metric: "std"}))
    archs_in_df = df["arch"].unique()
    for ax, opt in zip(axes, OPTIMIZERS):
        ax.set_title(OPT_LABELS[opt], fontsize=10, fontweight="bold")
        ax.axvspan( 9.5, 10.5, color="gold",  alpha=0.20, zorder=0)
        ax.axvspan(19.5, 20.5, color="coral", alpha=0.18, zorder=0)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
        for arch in archs_in_df:
            cfg_match = next(
                (v for v in ARCH_REGISTRY.values() if v["name"] == arch), {}
            )
            col = cfg_match.get("color", "#888888")
            lbl = cfg_match.get("label", arch)
            sub_m = mean_df[(mean_df["arch"] == arch) &
                            (mean_df["optimizer"] == opt)].sort_values("lag")
            sub_s = std_df[ (std_df["arch"]  == arch) &
                            (std_df["optimizer"] == opt)].sort_values("lag")
            if sub_m.empty:
                continue
            x  = sub_m["lag"].values
            y  = sub_m[metric].values
            sd = sub_s["std"].values if len(sub_s) == len(sub_m) \
                 else np.zeros_like(y)
            ax.plot(x, y, color=col, linewidth=1.8, label=lbl, zorder=2)
            ax.fill_between(x, y - sd, y + sd, color=col,
                            alpha=0.13, zorder=1)
        ax.set_xlabel("Lag tau (epochs)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0, max_lag)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Figure saved: {out_path.name}")
def _plot_combined(all_df, out_dir, max_lag):
    """Combined optimizer-panel plots across all completed stages."""
    combined_dir = out_dir / "combined"
    combined_dir.mkdir(exist_ok=True)
    _plot_per_optimizer(
        all_df, "acf_mag",
        ylabel="ACF of ||Delta_t||",
        title="Step Magnitude ACF -- All Architectures",
        out_path=combined_dir / "acf_magnitude_combined.pdf",
        max_lag=max_lag,
    )
    _plot_per_optimizer(
        all_df, "cos_sim",
        ylabel="Mean cos(Delta_t, Delta_{t+tau})",
        title="Step Direction Cosine Similarity -- All Architectures",
        out_path=combined_dir / "acf_direction_combined.pdf",
        max_lag=max_lag,
    )
    all_df.to_csv(combined_dir / "acf_all_stages.csv", index=False)
    print(f"\n  Combined outputs written to: {combined_dir}")
# ---------------------------------------------------------------------------
# DEPENDENCE REPORT AND NEXT-STAGE SUGGESTION
# ---------------------------------------------------------------------------
def _window_summary(df):
    """
    Window-average cosine similarities per (arch, optimizer, seed):
      C1_10  = mean cos_sim lags 1-10
      C11_20 = mean cos_sim lags 11-20
      C1_20  = mean cos_sim lags 1-20
    """
    rows = []
    for (arch, opt, seed), grp in df.groupby(["arch", "optimizer", "seed"]):
        cos    = grp.set_index("lag")["cos_sim"]
        c1_10  = float(cos[cos.index.isin(range(1,  11))].mean())
        c11_20 = float(cos[cos.index.isin(range(11, 21))].mean())
        c1_20  = float(cos[cos.index.isin(range(1,  21))].mean())
        rows.append({"arch": arch, "optimizer": opt, "seed": seed,
                     "C1_10": c1_10, "C11_20": c11_20, "C1_20": c1_20})
    return pd.DataFrame(rows)
def _dependence_report(df, arch_label):
    """
    Diagnostic string: spot-lag table, window averages, pairwise diffs,
    next-stage suggestion. Written to dependence_report.txt.
    This is diagnostic only — does not establish recurrence.
    """
    check_lags = [1, 5, 10, 20]
    pivot = (
        df[df["lag"].isin(check_lags)]
          .groupby(["optimizer", "lag"])[["acf_mag", "cos_sim"]]
          .mean().round(4)
    )
    lines = [
        "",
        "=" * 70,
        f"TEMPORAL DEPENDENCE REPORT -- {arch_label}",
        "  Diagnostic only. Does not establish recurrence.",
        "=" * 70,
        "  {:18s}{}".format(
            "Optimizer",
            "".join(f"  lag={l:>2}(mag/cos)" for l in check_lags)
        ),
    ]
    for opt in OPTIMIZERS:
        row = f"  {OPT_LABELS[opt]:<18}"
        for lag in check_lags:
            try:
                r = pivot.loc[(opt, lag)]
                row += f"  {r['acf_mag']:+.3f}/{r['cos_sim']:+.3f}"
            except KeyError:
                row += f"  {'N/A':>13}"
        lines.append(row)
    # Window averages
    win      = _window_summary(df)
    win_mean = (win.groupby("optimizer")[["C1_10", "C11_20", "C1_20"]]
                   .mean().round(4))
    lines += ["", "  WINDOW AVERAGES (mean cos_sim across seeds):"]
    lines.append(f"  {'Optimizer':<18}  C[1:10]   C[11:20]  C[1:20]")
    for opt in OPTIMIZERS:
        if opt in win_mean.index:
            r = win_mean.loc[opt]
            lines.append(
                f"  {OPT_LABELS[opt]:<18}  "
                f"{r['C1_10']:+.4f}   {r['C11_20']:+.4f}   {r['C1_20']:+.4f}"
            )
        else:
            lines.append(f"  {OPT_LABELS[opt]:<18}  N/A")
    # Pairwise C[1:20] differences
    lines += ["", "  PAIRWISE C[1:20] DIFFERENCES (descriptive):"]
    opts_present = [o for o in OPTIMIZERS if o in win_mean.index]
    for i, o1 in enumerate(opts_present):
        for o2 in opts_present[i+1:]:
            diff = win_mean.loc[o1, "C1_20"] - win_mean.loc[o2, "C1_20"]
            lines.append(
                f"  {OPT_LABELS[o1]:<18} vs {OPT_LABELS[o2]:<18}"
                f"  diff={diff:+.4f}"
            )
    # Next-stage suggestion
    lines += [""]
    any_structure = any(
        abs(win_mean.loc[o, "C1_20"]) > NEXT_STAGE_COS_THRESHOLD
        for o in opts_present if o in win_mean.index
    )
    if any_structure:
        note = (
            f"NEXT-STAGE SUGGESTION: At least one optimizer shows "
            f"|C[1:20]| > {NEXT_STAGE_COS_THRESHOLD}.\n"
            "  Non-trivial directional persistence detected.\n"
            "  Running the next stage may provide useful comparison.\n"
            "  This is a workflow heuristic, not a statistical criterion."
        )
    else:
        note = (
            "NEXT-STAGE SUGGESTION: All |C[1:20]| near zero.\n"
            "  Step directions appear approximately exchangeable.\n"
            "  Further stages are optional. Use --force_next to proceed."
        )
    lines.append(f"  {note}")
    lines.append("=" * 70)
    return "\n".join(lines)
def _suggest_next_stage(df, force):
    """
    Workflow heuristic: proceed if any optimizer has |C[1:20]| > threshold.
    Not a statistical test. Always True if force=True.
    """
    if force:
        return True
    try:
        win      = _window_summary(df)
        win_mean = win.groupby("optimizer")["C1_20"].mean()
        return bool((win_mean.abs() > NEXT_STAGE_COS_THRESHOLD).any())
    except Exception:
        return True
# ---------------------------------------------------------------------------
# RESNET RAM CHECK
# ---------------------------------------------------------------------------
def _resnet_ram_check():
    """Estimate peak RAM for ResNet and compare to available memory."""
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
    except ImportError:
        return True, "  psutil not installed -- skipping RAM check."
    # Chunked direction ACF peak: two step-chunk buffers + dot_accum
    T_steps  = 199      # T=200, so 199 steps
    peak_gb  = (
        2 * T_steps * CHUNK_COLS * 8 / 1e9 +   # two (T_steps x chunk) bufs
        MAX_LAG * T_steps * 8 / 1e9             # dot_accum
    )
    if avail_gb < peak_gb + 1.5:
        return False, (
            f"  RAM: available={avail_gb:.1f} GB  "
            f"estimated peak={peak_gb:.2f} GB\n"
            f"  May cause swapping. Use --force_next to proceed anyway."
        )
    return True, (
        f"  RAM: available={avail_gb:.1f} GB  "
        f"estimated peak={peak_gb:.2f} GB  -- OK"
    )
# ---------------------------------------------------------------------------
# NOTEBOOK CONFIG  —  edit these if running inside Jupyter
# ---------------------------------------------------------------------------
# When running in a notebook, argparse cannot read sys.argv (Jupyter fills it
# with kernel arguments). Set your options here instead; they are used
# automatically when a notebook kernel is detected.
NOTEBOOK_STAGES = ["mlp2"]          # options: "mlp2" "smallcnn" "resnet18" "all"
NOTEBOOK_TRAJ_DIR   = Path("exp4_results/trajectories")
NOTEBOOK_OUT_DIR    = Path("acf_results")
NOTEBOOK_MAX_LAG    = 40
NOTEBOOK_SEEDS      = None               # None = auto-detect from filenames on disk
NOTEBOOK_FORCE_NEXT = False              # set True to skip stage suggestion
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _is_notebook() -> bool:
    """True when running inside a Jupyter / IPython kernel."""
    try:
        shell = get_ipython().__class__.__name__          # noqa: F821
        return shell in ("ZMQInteractiveShell",           # Jupyter notebook/lab
                         "TerminalInteractiveShell")      # IPython terminal
    except NameError:
        return False
def parse_args():
    p = argparse.ArgumentParser(
        description="Staged ACF analysis: MLP2 first, SmallCNN second, "
                    "ResNet only if feasible.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (command line):
  python acf_step_analysis.py                          # Stage 1 only (default)
  python acf_step_analysis.py --stages mlp2 smallcnn  # Stages 1 and 2
  python acf_step_analysis.py --stages all             # All three stages
  python acf_step_analysis.py --force_next             # Skip stage suggestion
Jupyter notebook:
  Edit the NOTEBOOK_* variables near the top of this file, then run the cell.
        """
    )
    p.add_argument(
        "--stages", nargs="+",
        choices=["mlp2", "smallcnn", "resnet18", "all"],
        default=["mlp2"],
        help="Which stages to run. Default: mlp2 only."
    )
    p.add_argument("--traj_dir", type=Path, default=TRAJ_DIR,
                   help="Directory containing .npy trajectory files")
    p.add_argument("--out_dir",  type=Path, default=OUT_DIR,
                   help="Output directory")
    p.add_argument("--max_lag",  type=int,  default=MAX_LAG,
                   help="Maximum ACF lag in epochs (default 40)")
    p.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=(
            "Training seeds. If omitted, auto-discovered from filenames."
        )
    )
    p.add_argument(
        "--force_next", action="store_true",
        help="Skip next-stage suggestion and RAM check."
    )
    if _is_notebook():
        # Jupyter kernel puts its own flags in sys.argv — ignore them entirely
        # and fall back to the NOTEBOOK_* config variables defined above.
        args = p.parse_args([])          # parse an empty list → all defaults
        args.stages     = NOTEBOOK_STAGES
        args.traj_dir   = NOTEBOOK_TRAJ_DIR
        args.out_dir    = NOTEBOOK_OUT_DIR
        args.max_lag    = NOTEBOOK_MAX_LAG
        args.seeds      = NOTEBOOK_SEEDS
        args.force_next = NOTEBOOK_FORCE_NEXT
        print("[Notebook mode] Using NOTEBOOK_* config variables.")
        print(f"  stages={args.stages}  traj_dir={args.traj_dir}  "
              f"max_lag={args.max_lag}  seeds={args.seeds}  "
              f"force_next={args.force_next}")
    else:
        # Normal CLI: parse_known_args so stray flags don't crash the script
        args, unknown = p.parse_known_args()
        if unknown:
            print(f"[Warning] Ignoring unrecognised arguments: {unknown}")
    return args
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    # Expand "all" and deduplicate while preserving order
    raw = args.stages
    if "all" in raw:
        raw = ["mlp2", "smallcnn", "resnet18"]
    seen, ordered = set(), []
    for s in ["mlp2", "smallcnn", "resnet18"]:
        if s in raw and s not in seen:
            ordered.append(s)
            seen.add(s)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("STEP ACF ANALYSIS  --  staged by architecture")
    print(f"  Trajectory dir : {args.traj_dir}")
    print(f"  Output dir     : {args.out_dir}")
    print(f"  Max lag        : {args.max_lag} epochs")
    print(f"  Seeds          : {args.seeds}")
    print(f"  Stages to run  : {ordered}")
    print(f"  Force-next     : {args.force_next}")
    print("=" * 70)
    all_dfs  = []
    prev_df  = None
    for arch_key in ordered:
        cfg = ARCH_REGISTRY[arch_key]
        # -- Next-stage suggestion (workflow heuristic, not a test) ---------
        if cfg["stage"] > 1 and prev_df is not None and not args.force_next:
            if not _suggest_next_stage(prev_df, force=False):
                print(
                    f"\n  NEXT-STAGE SUGGESTION: All |C[1:20]| near zero in "
                    f"Stage {cfg['stage']-1}.\n"
                    f"  Proceeding to Stage {cfg['stage']} ({cfg['label']}) "
                    f"is optional.\n"
                    "  Inspect dependence_report.txt first.\n"
                    "  Use --force_next to run regardless. Stopping here."
                )
                break
        # -- ResNet RAM check ---------------------------------------------
        if arch_key == "resnet18":
            feasible, ram_msg = _resnet_ram_check()
            print(f"\n  ResNet-18 RAM estimate:\n{ram_msg}")
            if not feasible and not args.force_next:
                print(
                    "  Skipping ResNet-18 (insufficient RAM).\n"
                    "  Re-run with --force_next to proceed anyway.\n"
                )
                break
        # -- Run stage ----------------------------------------------------
        stage_df = analyse_arch(
            arch_key = arch_key,
            traj_dir = args.traj_dir,
            out_dir  = args.out_dir,
            seeds    = args.seeds,
            max_lag  = args.max_lag,
        )
        if not stage_df.empty:
            all_dfs.append(stage_df)
            prev_df = stage_df
    # -- Combined plots if multiple stages completed ----------------------
    if len(all_dfs) > 1:
        _plot_combined(pd.concat(all_dfs, ignore_index=True),
                       args.out_dir, args.max_lag)
    # -- Summary ----------------------------------------------------------
    completed = [
        ARCH_REGISTRY[k]["label"] for k in ordered
        if (args.out_dir / ARCH_REGISTRY[k]["subdir"]
            / "acf_summary.csv").exists()
    ]
    print("\n" + "=" * 70)
    print("COMPLETED STAGES:", completed if completed else "none")
    print()
    print("KEY OUTPUT FOR PAPER (Sec 6.7):")
    for arch_key in ordered:
        fig_path = (args.out_dir / ARCH_REGISTRY[arch_key]["subdir"]
                    / "acf_block_overlap.pdf")
        if fig_path.exists():
            print(f"    {fig_path}")
    if "mlp2" in ordered and "smallcnn" not in ordered:
        print()
        print("NEXT STEPS:")
        print("  1. Inspect stage1_mlp2/dependence_report.txt")
        print("  2. Inspect stage1_mlp2/acf_window_summary.csv")
        print("     (columns: C1_10, C11_20, C1_20 per optimizer)")
        print("  3. If structure visible: --stages mlp2 smallcnn")
    elif "smallcnn" in ordered and "resnet18" not in ordered:
        print()
        print("NEXT STEPS:")
        print("  1. Inspect stage2_smallcnn/dependence_report.txt")
        print("  2. Compare window CSVs across stages")
        print("  3. If consistent and RAM OK: --stages all")
    print("=" * 70)
if __name__ == "__main__":
    main()
