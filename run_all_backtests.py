"""
run_all_backtests.py — Run quintile-sort backtests for all factors in the registry.
===================================================================================

Reads backtest_data/factor_registry.csv to discover all factors, then runs
each one through the framework and produces:
  - Console summary for each factor
  - Individual PNG plot per factor
  - A combined comparison table (all_factors_summary.csv)

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

N_BUCKETS = 5   # quintiles (change to 3 for terciles with 15 countries)

# ---- Load registry -------------------------------------------------------

registry = pd.read_csv(DATA_DIR / "factor_registry.csv")
print(f"Found {len(registry)} factors in registry.\n")

# ---- Run all backtests ---------------------------------------------------

all_summaries = []

for _, row in registry.iterrows():
    factor_id   = row["factor_id"]
    label       = row["label"]
    higher      = bool(row["higher_is_better"])
    factor_file = DATA_DIR / row["filename"]
    returns_file = DATA_DIR / row["returns_file"]

    # Check files exist
    if not factor_file.exists():
        print(f"SKIPPING {label}: {factor_file} not found")
        continue
    if not returns_file.exists():
        print(f"SKIPPING {label}: {returns_file} not found")
        continue

    print("=" * 72)
    print(f"  Running: {label}  (higher_is_better={higher})")
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

    # Print summary
    result.print_summary()

    # Save plot
    fig = result.plot_all()
    plot_path = OUTPUT_DIR / f"{factor_id}.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}\n")

    # Collect summary row for comparison table
    summary = result.summary_table()
    ic = result.rank_ic()

    all_summaries.append({
        "Factor": label,
        "factor_id": factor_id,
        "higher_is_better": higher,
        "B1 Ann. Ret": summary.loc["B1", "Ann. Return"],
        f"B{N_BUCKETS} Ann. Ret": summary.iloc[N_BUCKETS - 1]["Ann. Return"],
        "L/S Ann. Ret": summary.iloc[-1]["Ann. Return"],
        "L/S Sharpe": summary.iloc[-1]["Sharpe"],
        "L/S Max DD": summary.iloc[-1]["Max DD"],
        "Monotonicity": result.monotonicity_score(),
        "Mean IC": ic.mean() if len(ic) > 0 else np.nan,
        "IC t-stat": (
            ic.mean() / ic.std() * np.sqrt(len(ic))
            if len(ic) > 1 else np.nan
        ),
        "Periods": len(result.bucket_returns),
        "Turnover": result.turnover().mean(),
    })


# ---- Comparison table ----------------------------------------------------

if all_summaries:
    comp = pd.DataFrame(all_summaries)

    # Sort by L/S Sharpe descending
    comp = comp.sort_values("L/S Sharpe", ascending=False)

    # Save
    comp_path = OUTPUT_DIR / "all_factors_summary.csv"
    comp.to_csv(str(comp_path), index=False)

    # Pretty-print
    print("\n" + "=" * 72)
    print("  FACTOR COMPARISON (sorted by L/S Sharpe)")
    print("=" * 72 + "\n")

    display = comp[[
        "Factor", "B1 Ann. Ret", "L/S Ann. Ret", "L/S Sharpe",
        "Monotonicity", "Mean IC", "IC t-stat", "Periods"
    ]].copy()

    for c in ["B1 Ann. Ret", "L/S Ann. Ret"]:
        display[c] = display[c].map(lambda x: f"{x:+.1%}")
    display["L/S Sharpe"] = display["L/S Sharpe"].map(lambda x: f"{x:.2f}")
    display["Monotonicity"] = display["Monotonicity"].map(lambda x: f"{x:.0%}")
    display["Mean IC"] = display["Mean IC"].map(lambda x: f"{x:.3f}")
    display["IC t-stat"] = display["IC t-stat"].map(lambda x: f"{x:.1f}")
    display["Periods"] = display["Periods"].astype(int)

    print(display.to_string(index=False))
    print(f"\nFull results saved to: {comp_path}")
    print(f"Individual plots in: {OUTPUT_DIR}/")
else:
    print("No backtests completed successfully.")
