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
# RESNET-SCALE POSITIVE CONTROL — CLEAN LOOP (LAMBDA=0)
# =============================================================================
#
# Purpose:
#   Test whether the graph-geodesic matched-step audit detects a clean
#   macroscopic loop embedded at ResNet-18 parameter scale.
#
# Decision convention:
#   formal_trigger = (pperm < alpha). nominal_trigger and separation/fallback
#   fields are retained as diagnostics only.
#
# Null convention:
#   Null trajectories are generated from the same loop-inclusive increments
#   used to define the observed trajectory, matching the matched-step null in
#   the manuscript.
#
# Formal geodesic runs use sentinel_fill=False: disconnected kNN graphs raise
# rather than being silently converted into finite distances.
# =============================================================================
# ====================== CLI CONFIG ======================
import argparse

def _as_bool(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}

def parse_args():
    p = argparse.ArgumentParser(description="ResNet-scale positive control (lambda=0 clean loop / optional drift)")
    p.add_argument("--traj-path", type=Path, default=Path("exp4_results/trajectories/resnet18_cifar10__sgd_momentum__seed20142.npy"))
    p.add_argument("--drift-scale", type=float, default=0.0)
    p.add_argument("--amp-fracs", default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2",
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
GEODESIC_K   = 12
# Formal-run policy: allow adaptive k to reach full T=200 checkpoint
# connectivity without sentinel fill.
GEODESIC_MAX_K = 199
GEODESIC_PCT = 95.0
ALPHA        = 0.05
EPS      = 1e-12
EPS_SIG  = 1e-6
MAD_SCALE = 1.4826
# Separation thresholds — diagnostics only.
# These constants have no theoretical status (Section 4.4 of the paper).
DELTA_MIN = 1e-3
LOBS_MULT = 5.0
try:
    import torch
    CUDA_OK = torch.cuda.is_available()
    DEVICE  = torch.device("cuda" if CUDA_OK else "cpu")
except Exception:
    CUDA_OK = False
    DEVICE  = None
print(f"Trajectory  : {traj_path}")
print(f"Drift scale : {drift_scale}  (0.0 = pure centered circle)")
print(f"FORMAL_NULLS: {FORMAL_NULLS}  (block_size=1, min pperm = 1/{FORMAL_NULLS+1} ≈ {1/(FORMAL_NULLS+1):.4f})")
print(f"BLOCK_NULLS : {BLOCK_NULLS}   (block_size>1, diagnostic only)")
print(f"TARGETED_MODE: {TARGETED_MODE}")
print(f"CUDA        : {CUDA_OK}")
# ====================== HELPERS ======================
def mad(a):
    med = np.median(a)
    return float(np.median(np.abs(a - med)))
def make_sparse_orthonormal_basis(d, support_size, seed):
    rng          = np.random.default_rng(seed)
    support_size = min(support_size, d)
    support      = rng.choice(d, size=support_size, replace=False)
    support.sort()
    u  = rng.standard_normal(support_size).astype(np.float32)
    u /= np.linalg.norm(u) + EPS
    v  = rng.standard_normal(support_size).astype(np.float32)
    v -= np.dot(v, u) * u
    v /= np.linalg.norm(v) + EPS
    return support.astype(np.int64), u, v
def loop_coefficients(n):
    """
    Return (cos_t, sin_t) for n equally-spaced angles in [0, 2π).
    With endpoint=False, this produces T-1 arc-steps of IDENTICAL
    magnitude (step_norm = 2*sin(π/T) ≈ 0.0314 for T=200).  The net
    displacement — the sum of all T-1 steps — equals Y[T-1] − Y[0], which is
    INVARIANT to permutation order (vector addition is commutative).  Every
    permuted null trajectory therefore starts and ends at the same two points
    as the observed trajectory.  This means:
      - Null trajectories are random tangles of equal-length steps, not random walks.
      - The clean circle's geodesic H1 (≈1.32) is expected to far exceed all null H1.
      - With n_nulls=200 we expect 0/200 nulls to beat Lobs → pperm ≈ 1/201 ≈ 0.005.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
def log_step_invariant(cos_t, sin_t):
    """
    Log the step-invariant properties at runtime for transparency.
    """
    cos_steps = np.diff(cos_t.astype(np.float64))
    sin_steps = np.diff(sin_t.astype(np.float64))
    step_norms = np.sqrt(cos_steps**2 + sin_steps**2)
    net_disp   = np.sqrt(cos_steps.sum()**2 + sin_steps.sum()**2)
    print(f"\n[Step-invariant check]")
    print(f"  Arc-step magnitude:  min={step_norms.min():.6f}  max={step_norms.max():.6f}  "
          f"std={step_norms.std():.2e}  (uniform: {np.allclose(step_norms, step_norms[0])})")
    print(f"  Net displacement of all {len(cos_steps)} steps: {net_disp:.6e}  "
          f"(invariant to permutation)")
def add_loop_to_chunk_inplace(Xc, col_start, support, u, v, cos_t, sin_t, amp):
    col_end = col_start + Xc.shape[1]
    left    = np.searchsorted(support, col_start)
    right   = np.searchsorted(support, col_end)
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
def accumulate_distance_from_chunks(chunks_iter, n):
    sq   = np.zeros(n, dtype=np.float64)
    gram = np.zeros((n, n), dtype=np.float64)
    for Xc in chunks_iter:
        if CUDA_OK:
            Xt    = torch.from_numpy(np.ascontiguousarray(Xc, dtype=np.float32)).to(DEVICE)
            sq   += (Xt * Xt).sum(dim=1).cpu().numpy().astype(np.float64)
            gram += torch.mm(Xt, Xt.t()).cpu().numpy().astype(np.float64)
            del Xt
            torch.cuda.empty_cache()
        else:
            X64   = Xc.astype(np.float64, copy=False)
            sq   += np.einsum("ij,ij->i", X64, X64)
            gram += X64 @ X64.T
        del Xc
        gc.collect()
    D2 = sq[:, None] + sq[None, :] - 2.0 * gram
    D  = np.sqrt(np.maximum(D2, 0.0))
    np.fill_diagonal(D, 0.0)
    return D
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
        context_label=context_label,
    )
    return L, sentinel
# ====================== SETUP ======================
X    = np.load(traj_path, mmap_mode="r")
n, d = X.shape
print(f"\nLoaded {n} x {d} trajectory")
cos_t, sin_t = loop_coefficients(n)
log_step_invariant(cos_t, sin_t)   # [Finding 2] runtime invariant check
# Median pairwise distance on the centered original trajectory.
# Used only to scale injected loop amplitude; not used for PH.
def base_iter():
    for start in range(0, d, chunk_cols):
        end      = min(start + chunk_cols, d)
        Xraw     = np.array(X[:, start:end], dtype=np.float32, copy=True)
        X0_chunk = np.array(X[0:1, start:end], dtype=np.float32, copy=True)
        yield Xraw - X0_chunk
D_base          = accumulate_distance_from_chunks(base_iter(), n)
median_pairwise = float(np.median(D_base[D_base > EPS]))
del D_base
gc.collect()
print(f"Median pairwise distance of centered trajectory = {median_pairwise:.6e}")
support, u, v = make_sparse_orthonormal_basis(d, support_size, basis_seed)
# ====================== MAIN SWEEP ======================
rows = []
for amp_frac in amp_fracs:
    amp = amp_frac * median_pairwise
    print()
    print("=" * 90)
    print(f"amp_frac={amp_frac:.3e}  amp={amp:.6e}  drift_scale={drift_scale}")
    print("=" * 90)
    # ------------------------------------------------------------------
    # Observed trajectory.
    #   drift_scale=0:  Y_t = injected circle only (pure H1 control).
    #   drift_scale=1:  Y_t = centered real trajectory + injected circle.
    # ------------------------------------------------------------------
    def obs_iter():
        for start in range(0, d, chunk_cols):
            end      = min(start + chunk_cols, d)
            Xraw     = np.array(X[:, start:end], dtype=np.float32, copy=True)
            X0_chunk = np.array(X[0:1, start:end], dtype=np.float32, copy=True)
            Xc       = drift_scale * (Xraw - X0_chunk)
            add_loop_to_chunk_inplace(Xc, start, support, u, v, cos_t, sin_t, amp)
            yield Xc
    D_obs      = accumulate_distance_from_chunks(obs_iter(), n)
    Lobs_geo, obs_sentinel = max_H1_geodesic_from_ambient(
        D_obs, context_label=f"(amp_frac={amp_frac:.1e}, observed)"
    )
    print(f"Observed Lobs_geo = {Lobs_geo:.6e}  sentinel={obs_sentinel}")
    del D_obs
    gc.collect()
    for b in block_sizes:
        # [M2]  Determine null count based on block size and targeted mode.
        if TARGETED_MODE:
            n_nulls = FORMAL_NULLS if b == 1 else BLOCK_NULLS
        else:
            n_nulls = FORMAL_NULLS
        min_pperm = 1.0 / (n_nulls + 1)
        is_formal = (b == 1)
        print(f"\n  block_size={b}  n_nulls={n_nulls}  "
              f"({'FORMAL — Type-I guaranteed' if is_formal else 'DIAGNOSTIC — no Type-I guarantee'})  "
              f"min_pperm={min_pperm:.4f}")
        Lnull_geo       = np.zeros(n_nulls)
        sentinel_counts = 0
        for i in range(n_nulls):
            rng  = np.random.default_rng([seed, i, b])
            perm = audit_block_permutation_indices(n - 1, b, rng)
            # The loop is added to Yc BEFORE computing steps,
            # so null steps are loop-inclusive increments of the combined
            # (drift + loop) trajectory.  This correctly implements the
            # matched-step null: each null is a random walk built from the
            # same increments as the observed trajectory, with temporal
            # order destroyed.  Section 4.3 of the paper covers this
            # implicitly; noted here for transparency.
            def null_iter():
                for start in range(0, d, chunk_cols):
                    end      = min(start + chunk_cols, d)
                    Xraw     = np.array(X[:, start:end], dtype=np.float32, copy=True)
                    X0_chunk = np.array(X[0:1, start:end], dtype=np.float32, copy=True)
                    Yc = drift_scale * (Xraw - X0_chunk)
                    add_loop_to_chunk_inplace(Yc, start, support, u, v, cos_t, sin_t, amp)
                    steps = Yc[1:] - Yc[:-1]   # loop-inclusive steps
                    steps = steps[perm]
                    Ynull    = np.empty_like(Yc)
                    Ynull[0] = Yc[0]
                    np.cumsum(steps, axis=0, out=Ynull[1:])
                    Ynull[1:] += Yc[0]
                    yield Ynull
            D_null = accumulate_distance_from_chunks(null_iter(), n)
            Lnull_geo[i], null_sent = max_H1_geodesic_from_ambient(
                D_null, context_label=f"(amp_frac={amp_frac:.1e}, null={i}, b={b})"
            )
            sentinel_counts += int(null_sent)
            del D_null
            gc.collect()
        # [M1]  compute_stats returns formal_trigger (pperm<alpha) separately
        # from nominal_trigger (legacy OR).  Use formal_trigger for inference.
        s = compute_stats(Lobs_geo, Lnull_geo)
        print(
            f"    pperm={s['pperm']:.4f}  "
            f"formal_trigger={s['formal_trigger']}  "   # [M1] primary
            f"fallback={s['fallback']}  "               # [M1] diagnostic
            f"nominal_trigger={s['nominal_trigger']}  " # [M1] diagnostic
            f"zrob={s['zrob']:+.2f}  "
            f"zf={s['zero_frac']:.2f}  "
            f"sentinel_nulls={sentinel_counts}/{n_nulls}"  # [Finding 5]
        )
        rows.append({
            "drift_scale":      drift_scale,
            "amp_frac":         amp_frac,
            "amp_abs":          amp,
            "block_size":       b,
            "n_nulls":          n_nulls,
            "is_formal_null":   is_formal,           # [M2] flag for reader
            "support_size":     support_size,
            "geodesic_k":       GEODESIC_K,
            "geodesic_pct":     GEODESIC_PCT,
            "Lobs_geo":         Lobs_geo,
            "obs_sentinel":     obs_sentinel,         # [Finding 5]
            "sentinel_nulls":   sentinel_counts,      # [Finding 5]
            **{f"geo_{k}": v for k, v in s.items()},
            "median_pairwise":  median_pairwise,
        })
        pd.DataFrame(rows).to_csv(out_file, index=False)
        print(f"    partial save → {out_file}")
# ====================== FINAL SUMMARY ======================
df = pd.DataFrame(rows)
# Save before validation checks, so failed validation still leaves
# an auditable CSV on disk.
df.to_csv(out_file, index=False)
print()
print("=" * 90)
print("FINAL RESULTS")
print("=" * 90)
# [M1]  Primary summary uses formal_trigger, not nominal_trigger.
#        fallback column included so readers can verify it never independently
#        drives a detection (mirrors Table-8 discipline of the paper).
print(df[[
    "drift_scale",
    "amp_frac",
    "block_size",
    "n_nulls",
    "is_formal_null",
    "Lobs_geo",
    "geo_pperm",
    "geo_formal_trigger",   # [M1] primary
    "geo_fallback",         # [M1] diagnostic
    "geo_nominal_trigger",  # [M1] diagnostic composite
    "geo_zrob",
    "geo_zero_frac",
    "sentinel_nulls",       # [Finding 5]
]].to_string(index=False))
# [M1]  Explicit check: flag any row where fallback fired without formal_trigger.
independent_fallbacks = df[df["geo_fallback"] & ~df["geo_formal_trigger"]]
if len(independent_fallbacks):
    print()
    print("WARNING: fallback fired independently of formal_trigger in the following cells:")
    print(independent_fallbacks[["amp_frac", "block_size", "geo_pperm", "geo_fallback"]])
    print("These cells have NO Type-I guarantee and should NOT be reported as detections.")
else:
    print()
    print("CHECK PASSED: fallback never fired independently of formal_trigger. "
          "All detections are pperm-driven (Type-I guaranteed).")
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
# Lambda=0 clean-loop positive-control check: formal b=1 rows should all fire.
if abs(drift_scale - 0.0) < 1e-12:
    formal_rows = df[df["is_formal_null"].astype(bool)]
    bad = formal_rows[~formal_rows["geo_formal_trigger"].astype(bool)]
    if len(bad):
        raise RuntimeError(
            "Lambda=0 ResNet-scale positive control failed: expected all "
            "formal b=1 rows to trigger.\n"
            + bad[[
                "amp_frac", "block_size", "n_nulls", "Lobs_geo",
                "geo_pperm", "geo_TSR", "geo_zero_frac",
            ]].to_string(index=False)
        )

print(f"\nSaved → {out_file}")
