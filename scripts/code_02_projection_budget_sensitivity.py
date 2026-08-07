#!/usr/bin/env python3
"""Projection-budget sensitivity audit.

Purpose
-------
Test whether the projection-level classification is stable as the random
projection budget varies over B in {5, 10, 20, 40, 100}.

Decision convention
-------------------
The observed and null statistics use the same projection budget. Formal
triggers are defined by pperm < ALPHA; TSR, zrob, delta, and separation fields
are descriptive only.

Outputs
-------
Writes results_corrected_long.csv and results_corrected_wide.csv under
results/projection_budget_sensitivity/.
"""
# Shared audit primitives.
from pathlib import Path as _AuditPath
import sys as _audit_sys
_audit_sys.path.insert(0, str(_AuditPath(__file__).resolve().parents[1]))
from audit_common import (
    compute_stats,
    safe_lifetime,
    matched_step_nulls as null_matched_steps,
)
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from ripser import ripser
from scipy.spatial.distance import pdist
import time
# ============================================================
# CONFIG
# ============================================================
OUT_DIR = Path("results") / "projection_budget_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 42
N_SAMPLES = 1200
NOISE_LEVEL = 0.03
TOTAL_PROJECTIONS = 100
BUDGETS = [5, 10, 20, 40, 100]
N_NULLS = 300
SUBSAMPLE = 600
ALPHA = 0.05
DELTA_MIN = 1e-3
LOBS_MULT = 5.0
EPS = 1e-12
EPS_SIG = 1e-6
MAD_SCALE = 1.4826
N_JOBS = -1
IS_PUB = (
    N_NULLS==300 and
    SUBSAMPLE==600 and
    TOTAL_PROJECTIONS==100
)
print("="*70)
print("Projection-budget sensitivity audit")
print(f"N_NULLS={N_NULLS}  SUBSAMPLE={SUBSAMPLE}  PROJECTIONS={TOTAL_PROJECTIONS}")
print(f"Reference configuration: {IS_PUB}")
print("="*70)
# ============================================================
# STATISTICS
# ============================================================
def _mad(arr):
    return float(np.median(np.abs(arr-np.median(arr))))
# ============================================================
# PH CORE
# ============================================================
def adaptive_thresh(X):
    d=pdist(X)
    if len(d)==0:
        return None
    return float(np.percentile(d,95))
def max_H1_euc(X,subsample,rng):
    if len(X)>subsample:
        X=X[rng.choice(len(X),subsample,replace=False)]
    thresh=adaptive_thresh(X)
    kwargs={"maxdim":1}
    if thresh is not None:
        kwargs["thresh"]=thresh
    dgms=ripser(X,**kwargs)["dgms"]
    return safe_lifetime(dgms, censoring_threshold=thresh)
# ============================================================
# NULL MODEL
# ============================================================
# ============================================================
# DETRENDING
# ============================================================
def detrend_pca(X):
    pca=PCA(n_components=1)
    return X-pca.inverse_transform(pca.fit_transform(X))
def detrend_time(X,time_vec):
    t=time_vec.reshape(-1,1)
    D=np.hstack([t,np.ones_like(t)])
    return X-D@np.linalg.lstsq(D,X,rcond=None)[0]
# ============================================================
# DATA GENERATORS
# ============================================================
def generate_linear(n,noise,rng):
    t=np.linspace(0,1,n)
    v=rng.standard_normal(3)
    v/=np.linalg.norm(v)
    X=np.outer(t*5.0,v)
    X+=noise*rng.standard_normal((n,3))
    return X,t
def generate_helix(n,noise,rng):
    t=np.linspace(0,1,n)
    th=6*np.pi*t
    X=np.vstack([
        3*np.cos(th),
        3*np.sin(th),
        5*t
    ]).T
    X+=noise*rng.standard_normal(X.shape)
    return X,t
def generate_circle(n,noise,rng):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    X=np.vstack([
        3*np.cos(t),
        3*np.sin(t),
        np.zeros_like(t)
    ]).T
    X+=noise*rng.standard_normal(X.shape)
    return X,t
# ============================================================
# NULL MATRIX
# ============================================================
def compute_null_matrix(nulls,proj_mats,subsample,seed_base):
    Ni=len(nulls)
    Np=len(proj_mats)
    flat=Parallel(n_jobs=N_JOBS)(
        delayed(max_H1_euc)(
            nulls[i]@proj_mats[j],
            subsample,
            np.random.default_rng(seed_base+i*1000+j)
        )
        for i in range(Ni)
        for j in range(Np)
    )
    return np.array(flat).reshape(Ni,Np)
# ============================================================
# SCENARIO RUNNER
# ============================================================
def run_scenario(name,generator,seed_offset):
    print(f"\nSCENARIO: {name}")
    X,tv=generator(
        N_SAMPLES,
        NOISE_LEVEL,
        np.random.default_rng(RNG_SEED+seed_offset)
    )
    # Projection family (fixed RNG bug)
    proj_rng=np.random.default_rng(RNG_SEED+seed_offset+1000)
    proj_mats=[]
    for _ in range(TOTAL_PROJECTIONS):
        Q,_=np.linalg.qr(
            proj_rng.standard_normal((3,2))
        )
        proj_mats.append(Q[:,:2])
    # Nulls
    nulls_orig=null_matched_steps(
        X,
        N_NULLS,
        np.random.default_rng(RNG_SEED+seed_offset+3000)
    )
    nulls_dpca=[detrend_pca(Xn) for Xn in nulls_orig]
    nulls_dtime=[detrend_time(Xn,tv) for Xn in nulls_orig]
    # Observed detrending
    X_dpca=detrend_pca(X)
    X_dtime=detrend_time(X,tv)
    def obs_L(Xt,seed_base):
        return np.array([
            max_H1_euc(
                Xt@proj_mats[i],
                SUBSAMPLE,
                np.random.default_rng(seed_base+i)
            )
            for i in range(TOTAL_PROJECTIONS)
        ])
    obs_raw=obs_L(X,RNG_SEED+seed_offset+2000)
    obs_dpca=obs_L(X_dpca,RNG_SEED+seed_offset+2100)
    obs_dtime=obs_L(X_dtime,RNG_SEED+seed_offset+2200)
    print("Computing null matrices...")
    H_raw=compute_null_matrix(
        nulls_orig,
        proj_mats,
        SUBSAMPLE,
        RNG_SEED+seed_offset+4000
    )
    H_dpca=compute_null_matrix(
        nulls_dpca,
        proj_mats,
        SUBSAMPLE,
        RNG_SEED+seed_offset+5000
    )
    H_dtime=compute_null_matrix(
        nulls_dtime,
        proj_mats,
        SUBSAMPLE,
        RNG_SEED+seed_offset+6000
    )
    results=[]
    for B in BUDGETS:
        s_raw=compute_stats(
            np.max(obs_raw[:B]),
            np.max(H_raw[:,:B],axis=1)
        )
        s_dpca=compute_stats(
            np.max(obs_dpca[:B]),
            np.max(H_dpca[:,:B],axis=1)
        )
        s_dtime=compute_stats(
            np.max(obs_dtime[:B]),
            np.max(H_dtime[:,:B],axis=1)
        )
        row=dict(
            Scenario=name,
            Budget=B,
            publication_config=IS_PUB
        )
        for k,v in s_raw.items():
            row["raw_"+k]=v
        for k,v in s_dpca.items():
            row["dpca_"+k]=v
        for k,v in s_dtime.items():
            row["dtime_"+k]=v
        results.append(row)
    return results
# ============================================================
# MAIN
# ============================================================
def main():
    t0=time.time()
    rows=[]
    rows+=run_scenario("Linear Drift",generate_linear,0)
    rows+=run_scenario("Helix",generate_helix,100)
    rows+=run_scenario("True Circle",generate_circle,200)
    df=pd.DataFrame(rows)
    out=OUT_DIR/"results_corrected_wide.csv"
    df.to_csv(out,index=False)

    # Long-form output for release table builders and manuscript auditing.
    long_rows=[]
    for _, row in df.iterrows():
        for method, prefix in [("raw", "raw"), ("detrended_pca", "dpca"), ("detrended_time", "dtime")]:
            out_row={
                "Scenario": row["Scenario"],
                "Budget": row["Budget"],
                "method": method,
                "publication_config": row["publication_config"],
                "N_SAMPLES": N_SAMPLES,
                "NOISE_LEVEL": NOISE_LEVEL,
                "TOTAL_PROJECTIONS": TOTAL_PROJECTIONS,
                "N_NULLS": N_NULLS,
                "SUBSAMPLE": SUBSAMPLE,
                "ALPHA": ALPHA,
                "DELTA_MIN": DELTA_MIN,
                "LOBS_MULT": LOBS_MULT,
            }
            for key in [
                "Lobs", "null_med", "null_max", "TSR", "pperm", "pct",
                "zrob", "delta", "zero_frac", "null_collapsed",
                "zrob_collapsed", "fallback", "separation_flag",
                "formal_trigger", "nominal_trigger",
            ]:
                col=f"{prefix}_{key}"
                if col in row.index:
                    out_row[key]=row[col]
            long_rows.append(out_row)
    long_df=pd.DataFrame(long_rows)
    long_out=OUT_DIR/"results_corrected_long.csv"
    long_df.to_csv(long_out,index=False)

    print("\nDone")
    print(f"Saved wide CSV: {out}")
    print(f"Saved long CSV: {long_out}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")
if __name__=="__main__":
    main()
