"""
run_all_backtests.py — Quintile-sort factor screening with eligibility checks
===============================================================================

1. Pre-screens every factor for data coverage (min 10 countries, min 36 periods)
2. Runs quintile sort only on eligible factors
3. Produces presentation-quality charts
4. Outputs a comparison table sorted by L/S Sharpe

Usage:
    python run_all_backtests.py
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from factor_backtest import (
    FactorBacktest, BacktestConfig, BacktestResult,
    check_factor_eligibility, apply_chart_style,
    PALETTE, BUCKET_COLORS,
)

apply_chart_style()

# =========================================================================
# Configuration
# =========================================================================

DATA_DIR   = Path("backtest_data")
OUTPUT_DIR = Path("backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

N_BUCKETS       = 5
MIN_COUNTRIES   = 10   # per period — need ≥2 per bucket
MIN_PERIODS     = 36   # at least 3 years of eligible data

# =========================================================================
# Load registry
# =========================================================================

registry = pd.read_csv(DATA_DIR / "factor_registry.csv")
print(f"Registry: {len(registry)} factors defined.\n")

# =========================================================================
# Phase 1: Eligibility screening
# =========================================================================

print("=" * 80)
print("  PHASE 1: DATA ELIGIBILITY SCREENING")
print(f"  Requirement: ≥{MIN_COUNTRIES} countries per date, "
      f"≥{MIN_PERIODS} eligible months")
print("=" * 80)

eligible_factors = []
ineligible_factors = []

for _, row in registry.iterrows():
    factor_id = row["factor_id"]
    label     = row["label"]
    category  = row.get("category", "")
    factor_file = DATA_DIR / row["filename"]

    if not factor_file.exists():
        print(f"  MISSING   {label:40s}  — file not found")
        ineligible_factors.append({"Factor": label, "Category": category,
                                   "Reason": "File not found"})
        continue

    info = check_factor_eligibility(
        factor_file, min_countries=MIN_COUNTRIES, min_eligible_periods=MIN_PERIODS)

    status = "ELIGIBLE" if info["eligible"] else "REJECTED"
    icon   = "✓" if info["eligible"] else "✗"

    print(f"  {icon} {status:10s} {label:40s}  "
          f"median={info['median_countries']:.0f} countries, "
          f"{info['eligible_dates']:3d} eligible months  "
          f"{'  — ' + info['reason'] if not info['eligible'] else ''}")

    if info["eligible"]:
        eligible_factors.append(row.to_dict() | info)
    else:
        ineligible_factors.append({
            "Factor": label, "Category": category, "Reason": info["reason"],
            "Max Countries": info["max_countries_in_data"],
        })

print(f"\n  Result: {len(eligible_factors)} eligible, "
      f"{len(ineligible_factors)} rejected\n")


# =========================================================================
# Phase 2: Run quintile-sort backtests on eligible factors
# =========================================================================

print("=" * 80)
print("  PHASE 2: QUINTILE SORT BACKTESTS")
print("=" * 80)

all_summaries = []
all_results = {}

for finfo in eligible_factors:
    factor_id = finfo["factor_id"]
    label     = finfo["label"]
    higher    = bool(finfo["higher_is_better"])
    category  = finfo.get("category", "Unknown")
    factor_file = DATA_DIR / finfo["filename"]

    config = BacktestConfig(
        n_buckets=N_BUCKETS,
        higher_is_better=higher,
        factor_name=label,
        min_countries=MIN_COUNTRIES,
        annualisation_factor=12,
    )

    bt = FactorBacktest(config)

    try:
        result = bt.run(
            factor_csv=str(factor_file),
            returns_csv=str(DATA_DIR / "returns.csv"),
        )
    except RuntimeError as e:
        print(f"\n  FAILED: {label} — {e}")
        continue

    all_results[factor_id] = result

    print(f"\n{'─' * 72}")
    result.print_summary()

    # Save individual plot
    fig = result.plot_all()
    fig.savefig(str(OUTPUT_DIR / f"{factor_id}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Collect summary
    summary = result.summary_table()
    ic = result.rank_ic()
    all_summaries.append({
        "Category": category,
        "Factor": label,
        "factor_id": factor_id,
        "higher_is_better": higher,
        "B1 Ann Ret": summary.loc["B1", "Ann. Return"],
        f"B{N_BUCKETS} Ann Ret": summary.iloc[N_BUCKETS - 1]["Ann. Return"],
        "L/S Ann Ret": summary.iloc[-1]["Ann. Return"],
        "L/S Ann Vol": summary.iloc[-1]["Ann. Vol"],
        "L/S Sharpe": summary.iloc[-1]["Sharpe"],
        "L/S Max DD": summary.iloc[-1]["Max DD"],
        "L/S Hit Rate": summary.iloc[-1]["Hit Rate"],
        "Monotonicity": result.monotonicity_score(),
        "Mean IC": ic.mean() if len(ic) > 0 else np.nan,
        "IC t-stat": (ic.mean() / ic.std() * np.sqrt(len(ic))
                      if len(ic) > 1 else np.nan),
        "IC Hit Rate": (ic > 0).mean() if len(ic) > 0 else np.nan,
        "Periods": len(result.bucket_returns),
        "Turnover": result.turnover().mean(),
    })


# =========================================================================
# Phase 3: Comparison table
# =========================================================================

if not all_summaries:
    print("No backtests completed.")
    exit()

comp = pd.DataFrame(all_summaries).sort_values("L/S Sharpe", ascending=False)
comp.to_csv(str(OUTPUT_DIR / "all_factors_summary.csv"), index=False)

print("\n\n" + "=" * 100)
print("  FACTOR COMPARISON — sorted by L/S Sharpe")
print(f"  (min {MIN_COUNTRIES} countries per period, "
      f"min {MIN_PERIODS} eligible periods)")
print("=" * 100 + "\n")

display = comp[[
    "Category", "Factor", "B1 Ann Ret", "L/S Ann Ret",
    "L/S Sharpe", "Monotonicity", "Mean IC", "IC t-stat", "Periods",
]].copy()

for c in ["B1 Ann Ret", "L/S Ann Ret"]:
    display[c] = display[c].map(lambda x: f"{x:+.1%}")
display["L/S Sharpe"] = display["L/S Sharpe"].map(lambda x: f"{x:.2f}")
display["Monotonicity"] = display["Monotonicity"].map(lambda x: f"{x:.0%}")
display["Mean IC"] = display["Mean IC"].map(lambda x: f"{x:.3f}")
display["IC t-stat"] = display["IC t-stat"].map(lambda x: f"{x:.1f}")
display["Periods"] = display["Periods"].astype(int)

print(display.to_string(index=False))

# Category summary
print("\n\n" + "=" * 80)
print("  CATEGORY SUMMARY")
print("=" * 80 + "\n")
cat_summary = comp.groupby("Category").agg({
    "L/S Sharpe": ["mean", "max", "count"],
    "Mean IC": "mean",
}).round(3)
cat_summary.columns = ["Avg Sharpe", "Best Sharpe", "# Factors", "Avg IC"]
cat_summary = cat_summary.sort_values("Avg Sharpe", ascending=False)
print(cat_summary.to_string())

# Ineligible factors
if ineligible_factors:
    print("\n\n" + "=" * 80)
    print("  INELIGIBLE FACTORS (excluded from backtest)")
    print("=" * 80 + "\n")
    inel_df = pd.DataFrame(ineligible_factors)
    print(inel_df.to_string(index=False))


# =========================================================================
# Phase 4: Presentation charts
# =========================================================================

# ---- Chart A: Annualised return by bucket, top 6 factors ----------------

top_n = min(6, len(all_summaries))
top_factors = comp.head(top_n)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Factor Screening — Annualised Returns by Quintile Bucket",
             fontsize=18, fontweight="bold", color=PALETTE["darkblue"], y=0.995)
fig.text(0.5, 0.965,
         f"Top {top_n} factors by L/S Sharpe  ·  "
         f"Min {MIN_COUNTRIES} countries per period  ·  "
         f"B1 = most attractive",
         ha="center", fontsize=11, color=PALETTE["grey"])

for idx, (_, row) in enumerate(top_factors.iterrows()):
    ax = axes.flat[idx]
    fid = row["factor_id"]
    result = all_results[fid]
    tbl = result.summary_table().iloc[:N_BUCKETS]

    colors = BUCKET_COLORS[:N_BUCKETS]
    bars = ax.bar(tbl.index.astype(str), tbl["Ann. Return"],
                  color=colors, edgecolor="white", linewidth=1.5,
                  width=0.6, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_title(f"{row['Factor']}\nSharpe={row['L/S Sharpe']:.2f}, "
                 f"Mono={row['Monotonicity']:.0%}",
                 fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    for bar, val in zip(bars, tbl["Ann. Return"]):
        offset = 0.003 if val >= 0 else -0.003
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:+.1%}", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")

# Hide unused axes
for idx in range(top_n, len(axes.flat)):
    axes.flat[idx].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(str(OUTPUT_DIR / "top_factors_bar.png"), dpi=200, bbox_inches="tight")
plt.close(fig)


# ---- Chart B: L/S cumulative for top factors ----------------------------

fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle("Long / Short Equity Curves — Top Factors",
             fontsize=16, fontweight="bold", color=PALETTE["darkblue"])

colors_cycle = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"],
                PALETTE["red"], "#8e44ad", "#16a085", "#2c3e50", "#d35400"]

for idx, (_, row) in enumerate(top_factors.iterrows()):
    fid = row["factor_id"]
    result = all_results[fid]
    cum_ls = result.cum_long_short
    label = f"{row['Factor']} ({row['L/S Sharpe']:.2f})"
    ax.plot(cum_ls.index, cum_ls.values, label=label,
            linewidth=2.0, color=colors_cycle[idx % len(colors_cycle)])

ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_ylabel("Growth of €1  (L/S)")
ax.legend(loc="upper left", frameon=True, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("€%.1f"))

plt.tight_layout()
fig.savefig(str(OUTPUT_DIR / "top_factors_ls_cumulative.png"),
            dpi=200, bbox_inches="tight")
plt.close(fig)


# ---- Chart C: Sharpe + IC summary bar chart -----------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Factor Signal Strength Summary",
             fontsize=16, fontweight="bold", color=PALETTE["darkblue"])

# Sort by L/S Sharpe for consistent ordering
plot_data = comp.sort_values("L/S Sharpe", ascending=True).tail(15)

# Sharpe bars
colors_sharpe = [PALETTE["green"] if v > 0 else PALETTE["red"]
                 for v in plot_data["L/S Sharpe"]]
ax1.barh(plot_data["Factor"], plot_data["L/S Sharpe"],
         color=colors_sharpe, edgecolor="white", linewidth=0.8, height=0.65)
ax1.axvline(0, color="black", linewidth=0.8)
ax1.set_xlabel("L/S Sharpe Ratio")
ax1.set_title("Long/Short Sharpe")
for i, (_, row) in enumerate(plot_data.iterrows()):
    ax1.text(row["L/S Sharpe"] + 0.02 * np.sign(row["L/S Sharpe"]),
             i, f"{row['L/S Sharpe']:.2f}", va="center", fontsize=9)

# IC t-stat bars
colors_ic = [PALETTE["blue"] if v > 1.96 else
             PALETTE["green"] if v > 0 else
             PALETTE["red"]
             for v in plot_data["IC t-stat"].fillna(0)]
ax2.barh(plot_data["Factor"], plot_data["IC t-stat"].fillna(0),
         color=colors_ic, edgecolor="white", linewidth=0.8, height=0.65)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.axvline(1.96, color=PALETTE["blue"], linewidth=1, linestyle="--",
            alpha=0.5, label="t=1.96 (5% sig.)")
ax2.set_xlabel("IC t-statistic")
ax2.set_title("Information Coefficient Significance")
ax2.legend(fontsize=9)

plt.tight_layout()
fig.savefig(str(OUTPUT_DIR / "factor_signal_strength.png"),
            dpi=200, bbox_inches="tight")
plt.close(fig)


# ---- Done ----------------------------------------------------------------

print(f"\n\nAll results saved to: {OUTPUT_DIR}/")
print(f"  all_factors_summary.csv            — comparison table")
print(f"  <factor_id>.png                    — individual 2×2 panels")
print(f"  top_factors_bar.png                — bar chart of top 6")
print(f"  top_factors_ls_cumulative.png      — L/S equity curves")
print(f"  factor_signal_strength.png         — Sharpe + IC summary")
