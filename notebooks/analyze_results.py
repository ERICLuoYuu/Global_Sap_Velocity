"""Analyze v3 benchmark results with pooled R² metrics."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CSV = os.environ.get(
    "BENCH_CSV",
    "outputs/statistics/gap_experiment/benchmark_results_full.csv",
)

df = pd.read_csv(CSV)
print(f"Total rows: {len(df):,}")
print("Methods: {}".format(df["method"].nunique()))
print("Has n_points: {}".format("n_points" in df.columns))
print()


def pooled_r2(g):
    v = g.dropna(subset=["n_points"])
    if v.empty:
        return pd.Series({"r2_pooled": np.nan, "rmse_pooled": np.nan, "mae_pooled": np.nan, "n_total": 0})
    N = v["n_points"].sum()
    if N == 0:
        return pd.Series({"r2_pooled": np.nan, "rmse_pooled": np.nan, "mae_pooled": np.nan, "n_total": 0})
    ss_res = v["ss_res"].sum()
    abs_err = v["sum_abs_err"].sum()
    s_true = v["sum_true"].sum()
    s_true_sq = v["sum_true_sq"].sum()
    mean_p = s_true / N
    ss_tot = s_true_sq - N * mean_p**2
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else (1.0 if ss_res < 1e-12 else 0.0)
    return pd.Series(
        {
            "r2_pooled": r2,
            "rmse_pooled": float(np.sqrt(ss_res / N)),
            "mae_pooled": float(abs_err / N),
            "n_total": int(N),
        }
    )


agg = df.groupby(["time_scale", "method", "group", "gap_size"]).apply(pooled_r2).reset_index()

SEP = "=" * 75

# ── Pooled R² by gap size ────────────────────────────────────────────────
for scale in ["hourly", "daily"]:
    sub = agg[agg.time_scale == scale]
    gap_sizes = sorted(sub["gap_size"].unique())

    print("\n" + SEP)
    print(f"  {scale.upper()} -- POOLED R2 by gap size (top 5 per gap)")
    print(SEP)
    for gs in gap_sizes:
        gsub = sub[sub.gap_size == gs].sort_values("r2_pooled", ascending=False).head(5)
        if scale == "hourly":
            label = f"{int(gs)}h" if gs < 24 else f"{int(gs // 24)}d"
        else:
            label = f"{int(gs)}d"
        parts = []
        for _, r in gsub.iterrows():
            parts.append("{} ({:.3f})".format(r["method"], r["r2_pooled"]))
        print("  gap={:<5} {}".format(label, "  ".join(parts)))

# ── Overall ranking ──────────────────────────────────────────────────────
for scale in ["hourly", "daily"]:
    sub = agg[agg.time_scale == scale]
    overall = (
        sub.groupby("method")
        .agg(
            r2_mean=("r2_pooled", "mean"),
            rmse_mean=("rmse_pooled", "mean"),
        )
        .sort_values("r2_mean", ascending=False)
        .reset_index()
    )
    print("\n" + SEP)
    print(f"  {scale.upper()} -- OVERALL RANKING (mean of pooled R2)")
    print(SEP)
    for _, row in overall.iterrows():
        r2v = row["r2_mean"]
        bar = "#" * max(0, int(r2v * 40)) if r2v > 0 else ""
        print("  {:<22} R2={:>7.4f}  RMSE={:>6.3f}  {}".format(row["method"], r2v, row["rmse_mean"], bar))

# ── DL vs non-DL ────────────────────────────────────────────────────────
print("\n" + SEP)
print("  HOURLY -- DL vs NON-DL (best pooled R2 per gap)")
print(SEP)
hourly = agg[agg.time_scale == "hourly"].copy()
hourly["is_dl"] = hourly["group"].isin(["D", "De"])
dl = hourly[hourly.is_dl].groupby("gap_size")["r2_pooled"].max()
nondl = hourly[~hourly.is_dl].groupby("gap_size")["r2_pooled"].max()
for gs in sorted(hourly["gap_size"].unique()):
    label = f"{int(gs)}h" if gs < 24 else f"{int(gs // 24)}d"
    d = dl.get(gs, np.nan)
    nd = nondl.get(gs, np.nan)
    winner = "DL" if d > nd else "non-DL"
    print(f"  gap={label:<5}  best_DL={d:.4f}  best_nonDL={nd:.4f}  winner={winner}")

# ── New methods ──────────────────────────────────────────────────────────
print("\n" + SEP)
print("  NEW vs EXISTING DL (hourly, mean pooled R2)")
print(SEP)
new_methods = ["D_bilstm", "D_gru", "De_bilstm_env", "De_gru_env"]
h_overall = hourly.groupby("method")["r2_pooled"].mean().sort_values(ascending=False)
for m in h_overall.index:
    if m.startswith("D"):
        tag = "[NEW]" if m in new_methods else "     "
        print(f"  {tag} {m:<22} mean_pooled_R2={h_overall[m]:.4f}")

# ── Small gap fix verification ───────────────────────────────────────────
print("\n" + SEP)
print("  SMALL GAP FIX (pooled vs old per-gap R2)")
print(SEP)
old_r2 = df[df.time_scale == "hourly"].groupby(["method", "gap_size"])["r2"].mean().reset_index()
for gs in [1, 2, 3, 6, 12, 24]:
    label = f"{gs}h"
    p_sub = agg[(agg.time_scale == "hourly") & (agg.gap_size == gs)]
    if p_sub.empty:
        continue
    best_p = p_sub.loc[p_sub["r2_pooled"].idxmax()]
    o_sub = old_r2[old_r2.gap_size == gs]
    best_o = o_sub.loc[o_sub["r2"].idxmax()]
    print(
        "  gap={:<4}  pooled: {} ({:.4f})  old: {} ({:.4f})".format(
            label, best_p["method"], best_p["r2_pooled"], best_o["method"], best_o["r2"]
        )
    )
