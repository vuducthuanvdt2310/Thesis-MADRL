#!/usr/bin/env python
"""
sample_size_calculation.py
==========================
Calculates the minimum number of simulation episodes (replications) required
to statistically prove GNN-HAPPO is superior to each baseline model, using
the Type I / Type II error (power-of-test) methodology.

Methodology
-----------
For a one-sided hypothesis test  H₀: μ_GNN ≥ μ_baseline  vs  H₁: μ_GNN < μ_baseline

The required sample size is:

    n = ⌈ ((Z_α + Z_β) · S_D / ε)² ⌉

where
    Z_α  = one-tailed critical value for Type I  error (significance level α)
    Z_β  = critical value          for Type II error (test power = 1 − β)
    S_D  = pooled std-dev of cost differences   (from pilot data)
    ε    = minimum detectable difference (practical threshold)

Parameters used
---------------
    α   = 0.05  → Z_α = 1.6449  (one-tailed)
    β   = 0.10  → Z_β = 1.2816  (power = 90 %)
    ε   = 30 000 (cost units — the minimum meaningful cost reduction)

References
----------
    Law & Kelton, "Simulation Modeling and Analysis", 4th ed., §9.5
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CSV paths  (same dict as Validate_Results.py)
# ─────────────────────────────────────────────────────────────────────────────
PATHS: dict[str, str] = {
    "GNN-HAPPO":      "evaluation_results/eval_gnn/results_gnn_happo.csv",
    "Standard-HAPPO": "evaluation_results/eval_base/results_standard_happo.csv",
    "MAPPO":          "evaluation_results/eval_mappo/results_mappo.csv",
    "S-s-Heuristic":  "evaluation_results/basestock_v1/results_ss_heuristic.csv",
}

METRIC = "Total_Cost"   # Column to use for the comparison

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Error-rate parameters
# ─────────────────────────────────────────────────────────────────────────────
ALPHA   = 0.05    # Type I  error  (false positive – wrongly claiming improvement)
BETA    = 0.10    # Type II error  (false negative – missing a true improvement)
EPSILON = 5_000  # Minimum detectable cost difference (practical threshold)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def load_costs(paths: dict[str, str]) -> dict[str, np.ndarray]:
    """Load Total_Cost column from each CSV; exit with a clear message on error."""
    costs = {}
    for name, path_str in paths.items():
        p = Path(path_str)
        if not p.exists():
            print(f"[ERROR] File not found for '{name}': {p}")
            print("        Run the corresponding test script first, or update PATHS.")
            sys.exit(1)
        df = pd.read_csv(p)
        if METRIC not in df.columns:
            print(f"[ERROR] Column '{METRIC}' missing in '{name}' CSV.")
            sys.exit(1)
        arr = df[METRIC].dropna().values.astype(float)
        costs[name] = arr
        print(f"  [OK] {name:<20}  {len(arr):>3d} pilot episodes loaded  ←  {p}")
    return costs


def pooled_std_of_differences(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pooled standard deviation of the pairwise differences D_i = a_i − b_i.

    When the pilot runs share a common random number seed (CRN), the
    *paired* S_D is the natural estimator.  When the lengths differ, we
    use only the min-length prefix so the pairing is well-defined, but
    also report the *unpooled* two-sample formula as a cross-check.

    S_D = std(D_i, ddof=1)
    """
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    return float(np.std(diff, ddof=1))


def required_replications(sd: float,
                           alpha: float = ALPHA,
                           beta:  float = BETA,
                           eps:   float = EPSILON) -> dict:
    """
    Calculate n via the power-of-test formula (one-tailed):

        n = ⌈ ((Z_α + Z_β) · S_D / ε)² ⌉

    Returns a dict with all intermediate values for easy reporting.
    """
    z_alpha = float(stats.norm.ppf(1 - alpha))        # one-tailed  (α = 0.05 → 1.6449)
    z_beta  = float(stats.norm.ppf(1 - beta))         # power level (β = 0.10 → 1.2816)
    raw_n   = ((z_alpha + z_beta) * sd / eps) ** 2
    n_ceil  = math.ceil(raw_n)
    return {
        "z_alpha": z_alpha,
        "z_beta":  z_beta,
        "S_D":     sd,
        "raw_n":   raw_n,
        "n":       n_ceil,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  SAMPLE-SIZE CALCULATION  —  Type I / Type II Error Methodology")
    print(f"{SEP}\n")

    # ── Error parameters ────────────────────────────────────────────────────
    z_a = stats.norm.ppf(1 - ALPHA)
    z_b = stats.norm.ppf(1 - BETA)
    print("  ┌─ Error Parameters ──────────────────────────────────────────┐")
    print(f"  │  α (Type I  / significance level)  = {ALPHA:.2f}                  │")
    print(f"  │  β (Type II / 1 − power)           = {BETA:.2f}  →  Power = {(1-BETA)*100:.0f} % │")
    print(f"  │  Z_α (one-tailed, 1−α)             = {z_a:.4f}               │")
    print(f"  │  Z_β (one-tailed, 1−β)             = {z_b:.4f}               │")
    print(f"  │  ε  (min. detectable difference)   = {EPSILON:,}             │")
    print(f"  │  Metric                            = {METRIC}             │")
    print("  └─────────────────────────────────────────────────────────────┘\n")

    # ── Load pilot data ─────────────────────────────────────────────────────
    print("  Loading pilot episode data …")
    costs = load_costs(PATHS)
    gnn_costs = costs["GNN-HAPPO"]
    print()

    # ── Per-baseline calculations ────────────────────────────────────────────
    print("─" * 70)
    print(f"  {'Baseline':<20}  {'S_D':>12}  {'raw n':>8}  {'n (ceil)':>9}  {'Sufficient?':>12}")
    print("─" * 70)

    pilot_n = len(gnn_costs)
    results = []

    for baseline_name, baseline_costs in costs.items():
        if baseline_name == "GNN-HAPPO":
            continue

        sd = pooled_std_of_differences(gnn_costs, baseline_costs)
        res = required_replications(sd)
        sufficient = "✓ YES" if pilot_n >= res["n"] else f"✗ need ≥ {res['n']}"
        results.append((baseline_name, sd, res["raw_n"], res["n"], sufficient))

        print(f"  {baseline_name:<20}  {sd:>12,.2f}  {res['raw_n']:>8.2f}  {res['n']:>9d}  {sufficient:>12}")

    print("─" * 70)

    # ── Worst-case recommendation ────────────────────────────────────────────
    max_n = max(r[3] for r in results)
    print(f"\n  ► RECOMMENDATION: Run at least  n = {max_n}  episodes per model")
    print(f"    (worst-case across all baseline comparisons)\n")

    # ── Formula reminder ────────────────────────────────────────────────────
    print("  Formula used (one-tailed power-of-test):")
    print("      n = ⌈ ((Z_α + Z_β) · S_D / ε)² ⌉")
    print(f"        = ⌈ (({z_a:.4f} + {z_b:.4f}) · S_D / {EPSILON:,})² ⌉\n")

    # ── Detailed per-baseline breakdown ─────────────────────────────────────
    print(f"{'─'*70}")
    print("  DETAILED BREAKDOWN")
    print(f"{'─'*70}")
    for baseline_name, sd, raw_n, n_ceil, sufficient in results:
        gnn_mu   = float(np.mean(gnn_costs))
        base_mu  = float(np.mean(costs[baseline_name]))
        delta    = base_mu - gnn_mu
        print(f"\n  GNN-HAPPO  vs  {baseline_name}")
        print(f"    Pilot episodes (n₀)              : {min(len(gnn_costs), len(costs[baseline_name]))}")
        print(f"    GNN-HAPPO  mean cost             : {gnn_mu:>15,.2f}")
        print(f"    {baseline_name:<20} mean cost  : {base_mu:>15,.2f}")
        print(f"    Observed mean difference (Δ)     : {delta:>+15,.2f}")
        print(f"    STD of differences        : {sd:>15,.2f}")
        print(f"    Required n (raw)                 : {raw_n:>15.4f}")
        # print(f"    Required n (⌈raw⌉)               : {n_ceil:>15d}")
        print(f"    number of episodeds needed   : {sufficient}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
