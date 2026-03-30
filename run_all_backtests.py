"""
run_all_backtests.py — Run quintile-sort backtests for all available factors.
=============================================================================

Reads backtest_data/factor_registry.csv, runs each factor through the
framework, and produces:
  - Console summary per factor
  - Individual PNG per factor  (in backtest_results/)
  - Combined comparison CSV    (all_factors_summary.csv)

Usage:
    python run_all_backtests.py
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from factor_backtest import FactorBacktest, BacktestConfig


# ---- Configuration -------------------------------------------------------

DATA_DIR   = Path("backtest_data")
OUTPUT_DIR = Path("backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

N_BUCKETS = 5   # quintiles

# ---- Load registry -------------------------------------------------------

registry = pd.read_csv(DATA_DIR / "factor_registry.csv")
print(f"Found {len(registry)} factors in registry.")
if "category" in registry.columns:
    for cat, grp in registry.groupby("category"):
        print(f"  {cat}: {len(grp)} factors")
print()

# ---- Run all backtests ---------------------------------------------------

all_summaries = []

for idx, row in registry.iterrows():
    factor_id   = row["factor_id"]
    label       = row["label"]
    higher      = bool(row["higher_is_better"])
    factor_file = DATA_DIR / row["filename"]
    returns_file = DATA_DIR / row["returns_file"]
    category    = row.get("category", "Unknown")

    if not factor_file.exists():
        print(f"SKIPPING {label}: {factor_file} not found")
        continue
    if not returns_file.exists():
        print(f"SKIPPING {label}: {returns_file} not found")
        continue

    print("=" * 72)
    print(f"  [{idx+1}/{len(registry)}]  {label}  "
          f"({category}, higher_is_better={higher})")
    print("=" * 72)

    config = BacktestConfig(
        n_buckets=N_BUCKETS,
        higher_is_better=higher,
        factor_name=label,
        annualisation_factor=12,
    )

    bt = FactorBacktest(config)

    try:
        result = bt.run(
            factor_csv=str(factor_file),
            returns_csv=str(returns_file),
        )
    except RuntimeError as e:
        print(f"  FAILED: {e}\n")
        continue

    result.print_summary()

    # Save plot
    fig = result.plot_all()
    plot_path = OUTPUT_DIR / f"{factor_id}.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}\n")

    # Collect summary
    summary = result.summary_table()
    ic = result.rank_ic()

    all_summaries.append({
        "Category": category,
        "Factor": label,
        "factor_id": factor_id,
        "higher_is_better": higher,
        "B1 Ann. Ret": summary.loc["B1", "Ann. Return"],
        f"B{N_BUCKETS} Ann. Ret": summary.iloc[N_BUCKETS - 1]["Ann. Return"],
        "L/S Ann. Ret": summary.iloc[-1]["Ann. Return"],
        "L/S Ann. Vol": summary.iloc[-1]["Ann. Vol"],
        "L/S Sharpe": summary.iloc[-1]["Sharpe"],
        "L/S Max DD": summary.iloc[-1]["Max DD"],
        "L/S Hit Rate": summary.iloc[-1]["Hit Rate"],
        "Monotonicity": result.monotonicity_score(),
        "Mean IC": ic.mean() if len(ic) > 0 else np.nan,
        "IC t-stat": (
            ic.mean() / ic.std() * np.sqrt(len(ic))
            if len(ic) > 1 else np.nan
        ),
        "IC Hit Rate": (ic > 0).mean() if len(ic) > 0 else np.nan,
        "Periods": len(result.bucket_returns),
        "Turnover": result.turnover().mean(),
    })


# ---- Comparison table ----------------------------------------------------

if all_summaries:
    comp = pd.DataFrame(all_summaries)
    comp = comp.sort_values("L/S Sharpe", ascending=False)

    # Save full results
    comp.to_csv(str(OUTPUT_DIR / "all_factors_summary.csv"), index=False)

    # Pretty-print
    print("\n" + "=" * 90)
    print("  FACTOR COMPARISON — sorted by L/S Sharpe")
    print("=" * 90 + "\n")

    display = comp[[
        "Category", "Factor", "B1 Ann. Ret", "L/S Ann. Ret",
        "L/S Sharpe", "Monotonicity", "Mean IC", "IC t-stat", "Periods"
    ]].copy()

    for c in ["B1 Ann. Ret", "L/S Ann. Ret"]:
        display[c] = display[c].map(lambda x: f"{x:+.1%}")
    display["L/S Sharpe"] = display["L/S Sharpe"].map(lambda x: f"{x:.2f}")
    display["Monotonicity"] = display["Monotonicity"].map(lambda x: f"{x:.0%}")
    display["Mean IC"] = display["Mean IC"].map(lambda x: f"{x:.3f}")
    display["IC t-stat"] = display["IC t-stat"].map(lambda x: f"{x:.1f}")
    display["Periods"] = display["Periods"].astype(int)

    print(display.to_string(index=False))

    # Category summary
    print("\n\n" + "=" * 90)
    print("  CATEGORY SUMMARY — average L/S Sharpe by factor category")
    print("=" * 90 + "\n")

    cat_summary = comp.groupby("Category").agg({
        "L/S Sharpe": ["mean", "max", "count"],
        "Mean IC": "mean",
    }).round(3)
    cat_summary.columns = ["Avg Sharpe", "Best Sharpe", "# Factors", "Avg IC"]
    cat_summary = cat_summary.sort_values("Avg Sharpe", ascending=False)
    print(cat_summary.to_string())

    print(f"\nFull results: {OUTPUT_DIR / 'all_factors_summary.csv'}")
    print(f"Plots:        {OUTPUT_DIR}/")

    # Flag highly correlated factor groups
    print("\n" + "-" * 90)
    print("  REMINDER: Many of these factors are correlated.")
    print("  Before treating them as independent signals, check cross-")
    print("  correlations. In particular, the 5+ value metrics will")
    print("  likely share >80% of their signal content.")
    print("-" * 90)

else:
    print("No backtests completed successfully.")
