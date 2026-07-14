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
    block_permutation_indices as audit_block_permutation_indices,
    geodesic_distance_matrix,
    geodesic_h1_lifetime,
    h1_lifetime_from_distance_matrix,
    safe_lifetime,
    sha256_file,
    stride_subsample as audit_stride_subsample,
)
# =============================================================================
# RESNET-SCALE POSITIVE CONTROL — REAL DRIFT PLUS LOOP (LAMBDA=1)
# =============================================================================
#
# Purpose:
#   Test whether a loop remains detectable after being superposed on a real
#   ResNet-18/SGD optimization trajectory.
#
# Decision convention:
#   formal_trigger = (pperm < alpha). nominal_trigger and separation/fallback
#   fields are retained as diagnostics only.
#
# Null convention:
#   Null trajectories are generated from the same drift-plus-loop increments
#   used to define the observed trajectory, matching the matched-step null in
#   the manuscript.
#
# Formal geodesic runs use sentinel_fill=False: disconnected kNN graphs raise
# rather than being silently converted into finite distances.
# =============================================================================
# ====================== CLI CONFIG ======================
import argparse
from pathlib import Path
import gc
import time
import numpy as np
import pandas as pd

def _as_bool(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}

def parse_args():
    p = argparse.ArgumentParser(description="ResNet-scale positive control (lambda=1 real drift + loop)")
    p.add_argument("--traj-path", type=Path, default=Path("exp4_results/trajectories/resnet18_cifar10__sgd__seed22042.npy"))
    p.add_argument("--drift-scale", type=float, default=1.0)
    p.add_argument("--amp-fracs", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1",
                   help="Comma-separated loop amplitude fractions.")
    p.add_argument("--block-sizes", default="1,5,10,20")
    p.add_argument("--formal-nulls", type=int, default=200)
    p.add_argument("--block-nulls", type=int, default=50)
    p.add_argument("--targeted-mode", default="true",
                   help="true: 200 nulls for b=1, 50 for b>1; false: 200 for all blocks.")
    p.add_argument("--support-size", type=int, default=4096)
    p.add_argument("--chunk-cols", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--basis-seed", type=int, default=2026)
    p.add_argument("--out-file", default=None)
    return p.parse_known_args()[0]

_args = parse_args()
traj_path = _args.traj_path
drift_scale = float(_args.drift_scale)
amp_fracs = [float(x) for x in str(_args.amp_fracs).split(",") if x]
block_sizes = [int(x) for x in str(_args.block_sizes).split(",") if x]
FORMAL_NULLS = int(_args.formal_nulls)
BLOCK_NULLS = int(_args.block_nulls)
TARGETED_MODE = _as_bool(_args.targeted_mode)
support_size = int(_args.support_size)
chunk_cols = int(_args.chunk_cols)
seed = int(_args.seed)
basis_seed = int(_args.basis_seed)
OUT_DIR = Path("results") / "resnet_positive_controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_file = (
    Path(_args.out_file)
    if _args.out_file is not None
    else OUT_DIR / f"positive_control_drift_{drift_scale:.2f}_corrected.csv"
)
out_file.parent.mkdir(parents=True, exist_ok=True)
# ====================== CONSTANTS ======================
GEODESIC_K     = 12
# Formal-run policy: allow adaptive k to reach full T=200 checkpoint
# connectivity without sentinel fill.
GEODESIC_MAX_K = 199
GEODESIC_PCT   = 95.0
ALPHA          = 0.05
EPS            = 1e-12
EPS_SIG        = 1e-6
MAD_SCALE      = 1.4826
DELTA_MIN      = 1e-3
LOBS_MULT      = 5.0
try:
    import torch
    CUDA_OK = torch.cuda.is_available()
    DEVICE  = torch.device("cuda" if CUDA_OK else "cpu")
except Exception:
    CUDA_OK = False
    DEVICE  = None
print(f"Trajectory   : {traj_path}")
print(f"Drift scale  : {drift_scale}  (real ResNet drift plus injected loop)")
print(f"FORMAL_NULLS : {FORMAL_NULLS}  (block_size=1, min pperm = 1/{FORMAL_NULLS+1} ≈ {1/(FORMAL_NULLS+1):.4f})")
print(f"BLOCK_NULLS  : {BLOCK_NULLS}   (block_size>1, diagnostic only)")
print(f"TARGETED_MODE: {TARGETED_MODE}")
print(f"CUDA         : {CUDA_OK}")
print()
print("Amplitude range note:")
print("  The tested amplitudes probe whether injected loops remain detectable under optimizer drift.")
print(f"  Amplitude fractions: {[f'{x:.0e}' for x in amp_fracs]}")
# ====================== HELPERS ======================
def mad(a):
    med = np.median(a)
    return float(np.median(np.abs(a - med)))
def make_sparse_orthonormal_basis(d, support_size, seed):
    rng          = np.random.default_rng(seed)
    support_size = min(support_size, d)
    support      = rng.choice(d, size=support_size, replace=False)
    support.sort()
    u  = rng.standard_normal(support_size).astype(np.float64)
    u /= np.linalg.norm(u) + EPS
    v  = rng.standard_normal(support_size).astype(np.float64)
    v -= np.dot(v, u) * u
    v /= np.linalg.norm(v) + EPS
    return support.astype(np.int64), u, v
def loop_coefficients(n):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.cos(theta).astype(np.float64), np.sin(theta).astype(np.float64)
def add_loop_to_chunk_inplace(Xc, col_start, support, u, v, cos_t, sin_t, amp):
    col_end     = col_start + Xc.shape[1]
    left        = np.searchsorted(support, col_start)
    right       = np.searchsorted(support, col_end)
    if right <= left:
        return
    cols_global = support[left:right]
    cols_local  = cols_global - col_start
    u_sub       = u[left:right]
    v_sub       = v[left:right]
    Xc[:, cols_local] += amp * (
        cos_t[:, None] * u_sub[None, :] +
        sin_t[:, None] * v_sub[None, :]
    )
def accumulate_step_gram_from_chunks(chunks_iter, n):
    """One float64 increment Gram pass over the high-dimensional trajectory."""
    G = np.zeros((n - 1, n - 1), dtype=np.float64)
    for Xc in chunks_iter:
        X64 = np.asarray(Xc, dtype=np.float64)
        steps = np.diff(X64, axis=0)
        G += steps @ steps.T
        del Xc, X64, steps
        gc.collect()
    return 0.5 * (G + G.T)


def distance_from_step_gram(G, perm=None):
    Gp = np.asarray(G, dtype=np.float64)
    if perm is not None:
        p = np.asarray(perm, dtype=int)
        Gp = Gp[np.ix_(p, p)]
    prefix = np.pad(np.cumsum(np.cumsum(Gp, axis=0), axis=1), ((1, 0), (1, 0)))
    diag = np.diag(prefix)
    D2 = diag[:, None] + diag[None, :] - prefix - prefix.T
    D2 = 0.5 * (D2 + D2.T)
    D = np.sqrt(np.maximum(D2, 0.0))
    np.fill_diagonal(D, 0.0)
    return D


def accumulate_distance_from_chunks(chunks_iter, n):
    return distance_from_step_gram(accumulate_step_gram_from_chunks(chunks_iter, n))
def max_H1_from_distance(D, pct=GEODESIC_PCT):
    return h1_lifetime_from_distance_matrix(D, pct=pct, eps=EPS, rescale=False)
# Formal release policy: disconnected kNN-geodesic graphs fail loudly.
# We keep the sentinel bookkeeping variable only for backward-compatible
# output messages; with sentinel_fill=False it must remain False unless a
# future diagnostic-only run explicitly changes the policy.
_sentinel_fired = False
def max_H1_geodesic_from_ambient(D_full, context_label=""):
    L, sentinel = geodesic_h1_lifetime(
        D_full,
        input_distance_matrix=True,
        k0=GEODESIC_K,
        kmax=GEODESIC_MAX_K,
        pct=GEODESIC_PCT,
        eps=EPS,
        sentinel_fill=False,
        return_sentinel=True,
        complete_filtration=True,
        context_label=context_label,
    )
    return L, sentinel
# ====================== SETUP ======================
X    = np.load(traj_path, mmap_mode="r")
n, d = X.shape
print(f"\nLoaded {n} x {d} trajectory")
cos_t, sin_t = loop_coefficients(n)
support, u, v = make_sparse_orthonormal_basis(d, support_size, basis_seed)
# Median pairwise distance on centered trajectory (for amplitude scaling only).
def base_iter():
    for start in range(0, d, chunk_cols):
        end      = min(start + chunk_cols, d)
        Xraw     = np.array(X[:, start:end], dtype=np.float64, copy=True)
        X0_chunk = np.array(X[0:1, start:end], dtype=np.float64, copy=True)
        yield Xraw - X0_chunk
D_base          = accumulate_distance_from_chunks(base_iter(), n)
median_pairwise = float(np.median(D_base[D_base > EPS]))
del D_base
gc.collect()
print(f"Median pairwise distance (centered) = {median_pairwise:.6e}")
# ====================== MAIN SWEEP ======================
rows = []
for amp_frac in amp_fracs:
    amp   = amp_frac * median_pairwise
    t_amp = time.time()
    print()
    print("=" * 90)
    print(f"amp_frac={amp_frac:.3e}  amp={amp:.6e}  drift_scale={drift_scale}  (REAL DRIFT + LOOP)")
    print("=" * 90)
    # ------------------------------------------------------------------
    # Observed trajectory: centered real drift + injected loop.
    # ------------------------------------------------------------------
    def obs_iter():
        for start in range(0, d, chunk_cols):
            end      = min(start + chunk_cols, d)
            Xraw     = np.array(X[:, start:end], dtype=np.float64, copy=True)
            X0_chunk = np.array(X[0:1, start:end], dtype=np.float64, copy=True)
            Xc       = drift_scale * (Xraw - X0_chunk)
            add_loop_to_chunk_inplace(Xc, start, support, u, v, cos_t, sin_t, amp)
            yield Xc
    step_gram_obs = accumulate_step_gram_from_chunks(obs_iter(), n)
    D_obs = distance_from_step_gram(step_gram_obs)
    Lobs_geo, obs_sentinel = max_H1_geodesic_from_ambient(
        D_obs, context_label=f"(amp_frac={amp_frac:.1e}, observed)"
    )
    print(f"Observed Lobs_geo = {Lobs_geo:.6e}  obs_sentinel={obs_sentinel}")
    del D_obs
    gc.collect()
    for b in block_sizes:
        # Null count depends on block size and targeted mode.
        if TARGETED_MODE:
            n_nulls = FORMAL_NULLS if b == 1 else BLOCK_NULLS
        else:
            n_nulls = FORMAL_NULLS
        is_formal = (b == 1)
        min_pperm = 1.0 / (n_nulls + 1)
        t_block   = time.time()
        print(f"\n  block_size={b}  n_nulls={n_nulls}  "
              f"({'FORMAL — Type-I guaranteed' if is_formal else 'DIAGNOSTIC — no Type-I guarantee'})  "
              f"min_pperm={min_pperm:.4f}")
        Lnull_geo       = np.zeros(n_nulls)
        sentinel_counts = 0
        for i in range(n_nulls):
            rng  = np.random.default_rng([seed, i, b])
            perm = audit_block_permutation_indices(n - 1, b, rng)
            # The same loop-inclusive increment Gram is reordered by the
            # block permutation; no high-dimensional null trajectory is rebuilt.
            D_null = distance_from_step_gram(step_gram_obs, perm)
            Lnull_geo[i], null_sent = max_H1_geodesic_from_ambient(
                D_null,
                context_label=f"(amp_frac={amp_frac:.1e}, null={i}, b={b})"
            )
            sentinel_counts += int(null_sent)
            del D_null
            gc.collect()
        # Use formal_trigger as the primary decision variable.
        s = compute_stats(Lobs_geo, Lnull_geo)
        print(
            f"    pperm={s['pperm']:.4f}  "
            f"formal_trigger={s['formal_trigger']}  "   # primary
            f"fallback={s['fallback']}  "               # diagnostic
            f"nominal_trigger={s['nominal_trigger']}  " # diagnostic
            f"zrob={s['zrob']:+.2f}  "
            f"zf={s['zero_frac']:.2f}  "
            f"sentinel_nulls={sentinel_counts}/{n_nulls}  "
            f"block_time={time.time()-t_block:.1f}s"
        )
        rows.append({
            "drift_scale":     drift_scale,
            "amp_frac":        amp_frac,
            "amp_abs":         amp,
            "block_size":      b,
            "n_nulls":         n_nulls,
            "is_formal_null":  is_formal,
            "analysis_role": ("formal_matched_step" if is_formal else "block_diagnostic"),
            "is_formal_inference": is_formal,
            "support_size":    support_size,
            "geodesic_k":      GEODESIC_K,
            "geodesic_pct":    GEODESIC_PCT,
            "Lobs_geo":        Lobs_geo,
            "obs_sentinel":    obs_sentinel,
            "sentinel_nulls":  sentinel_counts,
            **{f"geo_{k}": v for k, v in s.items()},
            "median_pairwise": median_pairwise,
        })
        pd.DataFrame(rows).to_csv(out_file, index=False)
        print(f"    partial save → {out_file}")
    print(f"  Amplitude finished in {time.time()-t_amp:.1f}s")
# Assemble CSV and summary output. Partial per-amplitude saves are retained
# during execution, and the final df.to_csv() is the manuscript-facing artifact.
# ====================== FINAL SUMMARY ======================
df = pd.DataFrame(rows)
# Save before validation checks, so failed validation still leaves
# an auditable CSV on disk.
df.to_csv(out_file, index=False)
print()
print("=" * 90)
print("FINAL RESULTS")
print("=" * 90)
# Primary summary uses formal_trigger; fallback is included for the audit trail.
print(df[[
    "drift_scale",
    "amp_frac",
    "block_size",
    "n_nulls",
    "is_formal_null",
    "Lobs_geo",
    "geo_pperm",
    "geo_formal_trigger",
    "geo_fallback",
    "geo_nominal_trigger",
    "geo_zrob",
    "geo_zero_frac",
    "sentinel_nulls",
]].to_string(index=False))
# Explicit check: any row where fallback fires without formal_trigger.
independent_fallbacks = df[df["geo_fallback"] & ~df["geo_formal_trigger"]]
if len(independent_fallbacks):
    print()
    print("WARNING: fallback fired independently of formal_trigger in the following cells:")
    print(independent_fallbacks[["amp_frac", "block_size", "geo_pperm", "geo_fallback"]])
    print("These cells have NO Type-I guarantee and must NOT be reported as detections.")
else:
    print()
    print("CHECK PASSED: fallback never fired independently of formal_trigger. "
          "All detections (if any) are pperm-driven (Type-I guaranteed).")
# Global sentinel check based on saved row columns.
any_sentinel = bool(
    df["obs_sentinel"].astype(bool).any()
    or (df["sentinel_nulls"].astype(int) > 0).any()
)
if any_sentinel:
    bad_sentinel = df[
        df["obs_sentinel"].astype(bool)
        | (df["sentinel_nulls"].astype(int) > 0)
    ]
    raise RuntimeError(
        "Geodesic sentinel was triggered despite sentinel_fill=False.\n"
        + bad_sentinel[[
            "drift_scale", "amp_frac", "block_size",
            "obs_sentinel", "sentinel_nulls",
        ]].to_string(index=False)
    )
else:
    print()
    print("CHECK PASSED: Geodesic sentinel never fired. "
          "All geodesic distances are based on connected k-NN graphs.")
# Lambda=1 drift-superposed control check: formal b=1 rows should not fire
# under the manuscript's claimed collapse/non-detection result. If this
# changes, the manuscript claim must be updated rather than silently reused.
if abs(drift_scale - 1.0) < 1e-12:
    formal_rows = df[df["is_formal_null"].astype(bool)]
    bad = formal_rows[formal_rows["geo_formal_trigger"].astype(bool)]
    if len(bad):
        raise RuntimeError(
            "Lambda=1 drift-superposed positive-control audit changed: "
            "formal b=1 trigger occurred. Update the manuscript or inspect outputs.\n"
            + bad[[
                "amp_frac", "block_size", "n_nulls", "Lobs_geo",
                "geo_pperm", "geo_TSR", "geo_zero_frac",
            ]].to_string(index=False)
        )

print(f"\nSaved → {out_file}")
