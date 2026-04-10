"""
run_all_backtests.py — Quintile-sort screening with daily equity curves
========================================================================

1. Pre-screens factors for data coverage
2. Runs daily backtests on eligible factors
3. Sorts by Information Ratio (B1 vs MSCI EM) — relevant for long-only
4. Produces comparison charts with common start dates
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from factor_backtest import (
    FactorBacktest, BacktestConfig, check_factor_eligibility,
    load_daily_prices, align_to_common_start,
    apply_chart_style, PALETTE, BUCKET_COLORS, BENCHMARK_NAME,
)

apply_chart_style()


# Configuration
DATA_DIR    = Path("backtest_data")
OUTPUT_DIR  = Path("backtest_results")
PRICE_FILE  = "EM_Indices_EUR.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

N_BUCKETS     = 5
MIN_COUNTRIES = 10
MIN_PERIODS   = 36


# ---- Load daily prices ONCE ----------------------------------------------
print("Loading daily price data...")
country_prices, benchmark_prices = load_daily_prices(PRICE_FILE)
print(f"  {country_prices.shape[1]} countries, "
      f"{country_prices.shape[0]} trading days")
print(f"  Date range: {country_prices.index.min().date()} to "
      f"{country_prices.index.max().date()}")
print(f"  Benchmark: {'OK' if benchmark_prices is not None else 'NOT FOUND'}\n")


# ---- Load registry --------------------------------------------------------
registry = pd.read_csv(DATA_DIR / "factor_registry.csv")
print(f"Registry: {len(registry)} factors defined.\n")


# =========================================================================
# Phase 1: Eligibility screening
# =========================================================================

print("=" * 80)
print("  PHASE 1: ELIGIBILITY SCREENING")
print(f"  Requirement: ≥{MIN_COUNTRIES} countries per date, "
      f"≥{MIN_PERIODS} eligible months")
print("=" * 80)

eligible_factors = []
ineligible_factors = []

for _, row in registry.iterrows():
    factor_id = row["factor_id"]
    label = row["label"]
    factor_file = DATA_DIR / row["filename"]

    if not factor_file.exists():
        print(f"  MISSING   {label}")
        continue

    info = check_factor_eligibility(factor_file, MIN_COUNTRIES, MIN_PERIODS)
    status = "[OK] ELIGIBLE" if info["eligible"] else "[--] REJECTED"

    print(f"  {status:14s} {label:40s}  "
          f"median={info['median_countries']:.0f} countries, "
          f"{info['eligible_dates']:3d} eligible months"
          f"{'  — ' + info['reason'] if not info['eligible'] else ''}")

    if info["eligible"]:
        eligible_factors.append({**row.to_dict(), **info})
    else:
        ineligible_factors.append({
            "Factor": label, "Reason": info["reason"],
            "Max Countries": info["max_countries_in_data"],
        })

print(f"\n  Result: {len(eligible_factors)} eligible, "
      f"{len(ineligible_factors)} rejected\n")


# =========================================================================
# Phase 2: Run backtests
# =========================================================================

print("=" * 80)
print("  PHASE 2: BACKTESTS (DAILY EQUITY CURVES)")
print("=" * 80)

all_summaries = []
all_results = {}

for finfo in eligible_factors:
    factor_id = finfo["factor_id"]
    label = finfo["label"]
    higher = bool(finfo["higher_is_better"])
    category = finfo.get("category", "")
    factor_file = DATA_DIR / finfo["filename"]

    config = BacktestConfig(
        n_buckets=N_BUCKETS,
        higher_is_better=higher,
        min_countries=MIN_COUNTRIES,
        factor_name=label,
        periods_per_year=252,
    )

    bt = FactorBacktest(config)

    try:
        result = bt.run(
            factor_csv=str(factor_file),
            daily_country_prices=country_prices,
            benchmark_daily_prices=benchmark_prices,
        )
    except Exception as e:
        print(f"\n  FAILED: {label} — {e}")
        continue

    all_results[factor_id] = result

    print(f"\n{'─' * 72}")
    result.print_summary()

    fig = result.plot_all()
    fig.savefig(str(OUTPUT_DIR / f"{factor_id}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Collect summary
    tbl = result.summary_table()
    ex = result.b1_excess_stats()
    ic = result.rank_ic()

    all_summaries.append({
        "Category": category,
        "Factor": label,
        "factor_id": factor_id,
        "higher_is_better": higher,
        "B1 Ann Ret": tbl.loc["B1", "Ann Return"],
        "B1 Vol": tbl.loc["B1", "Ann Vol"],
        "B1 Sharpe": tbl.loc["B1", "Sharpe"],
        "B1 Max DD": tbl.loc["B1", "Max DD"],
        f"B{N_BUCKETS} Ann Ret": tbl.loc[f"B{N_BUCKETS}", "Ann Return"],
        "BM Ann Ret": (tbl.loc[BENCHMARK_NAME, "Ann Return"]
                       if BENCHMARK_NAME in tbl.index else np.nan),
        "Excess vs BM": ex.get("Excess Return", np.nan),
        "Tracking Error": ex.get("Tracking Error", np.nan),
        "Information Ratio": ex.get("Information Ratio", np.nan),
        "Hit Rate vs BM": ex.get("Hit Rate vs BM", np.nan),
        "Monotonicity": result.monotonicity_score(),
        "Mean IC": ic.mean() if len(ic) > 0 else np.nan,
        "IC t-stat": (ic.mean() / ic.std() * np.sqrt(len(ic))
                      if len(ic) > 1 else np.nan),
        "Days": len(result.daily_bucket_returns),
    })


# =========================================================================
# Phase 3: Comparison table
# =========================================================================

if not all_summaries:
    print("No backtests completed.")
    raise SystemExit()

comp = pd.DataFrame(all_summaries).sort_values("Information Ratio", ascending=False)
comp.to_csv(str(OUTPUT_DIR / "all_factors_summary.csv"), index=False)

print("\n\n" + "=" * 110)
print(f"  FACTOR COMPARISON — sorted by Information Ratio (B1 vs {BENCHMARK_NAME})")
print(f"  Long-only top bucket  ·  Daily returns  ·  Min {MIN_COUNTRIES} countries")
print("=" * 110 + "\n")

display = comp[[
    "Category", "Factor", "B1 Ann Ret", "B1 Sharpe", "B1 Max DD",
    "BM Ann Ret", "Excess vs BM", "Information Ratio",
    "Monotonicity", "IC t-stat", "Days"
]].copy()

for c in ["B1 Ann Ret", "BM Ann Ret", "Excess vs BM", "B1 Max DD"]:
    display[c] = display[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")
display["B1 Sharpe"] = display["B1 Sharpe"].map(
    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
display["Information Ratio"] = display["Information Ratio"].map(
    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
display["Monotonicity"] = display["Monotonicity"].map(
    lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
display["IC t-stat"] = display["IC t-stat"].map(
    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
display["Days"] = display["Days"].astype(int)

print(display.to_string(index=False))


# =========================================================================
# Phase 4: Aligned comparison charts
# =========================================================================

top_n = min(6, len(comp))
top_factors = comp.head(top_n)

# ---- Chart A: Bar grid for top factors ----
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(f"Top {top_n} Factors — Annualised Return by Bucket",
             fontsize=18, fontweight="bold", color=PALETTE["darkblue"], y=0.995)
fig.text(0.5, 0.965,
         f"Sorted by Information Ratio vs {BENCHMARK_NAME}  ·  "
         f"B1 = most attractive  ·  Min {MIN_COUNTRIES} countries",
         ha="center", fontsize=11, color=PALETTE["grey"])

for idx, (_, row) in enumerate(top_factors.iterrows()):
    ax = axes.flat[idx]
    fid = row["factor_id"]
    result = all_results[fid]
    tbl = result.summary_table()
    bucket_rows = tbl[tbl.index.str.startswith("B")]

    colors = BUCKET_COLORS[:N_BUCKETS]
    bars = ax.bar(bucket_rows.index.astype(str), bucket_rows["Ann Return"],
                  color=colors, edgecolor="white", linewidth=1.5,
                  width=0.6, zorder=3)

    if BENCHMARK_NAME in tbl.index:
        bm_ret = tbl.loc[BENCHMARK_NAME, "Ann Return"]
        ax.axhline(bm_ret, color=PALETTE["benchmark"], linewidth=1.8,
                   linestyle="--", label=f"{BENCHMARK_NAME}")
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_title(f"{row['Factor']}\nIR={row['Information Ratio']:.2f}, "
                 f"Mono={row['Monotonicity']:.0%}", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    for bar, val in zip(bars, bucket_rows["Ann Return"]):
        offset = 0.003 if val >= 0 else -0.003
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:+.1%}", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")

for idx in range(top_n, len(axes.flat)):
    axes.flat[idx].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(str(OUTPUT_DIR / "top_factors_bar.png"), dpi=200, bbox_inches="tight")
plt.close(fig)


# ---- Chart B: B1 equity curves for top factors (aligned) ----
top_b1_curves = {}
for _, row in top_factors.iterrows():
    fid = row["factor_id"]
    result = all_results[fid]
    top_b1_curves[row["Factor"]] = result.bucket_equity["B1"]

if benchmark_prices is not None:
    bm_returns = benchmark_prices.pct_change()
    bm_equity = (1 + bm_returns.fillna(0)).cumprod()
    top_b1_curves[BENCHMARK_NAME] = bm_equity

aligned_curves = align_to_common_start(top_b1_curves)

fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle(f"B1 (Top Bucket) Equity Curves — Top {top_n} Factors",
             fontsize=16, fontweight="bold", color=PALETTE["darkblue"])

if aligned_curves:
    common_start = min(c.index[0] for c in aligned_curves.values())
    fig.text(0.5, 0.93,
             f"Aligned to common start: {common_start.date()}  ·  Daily data",
             ha="center", fontsize=10, color=PALETTE["grey"])

colors_cycle = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"],
                PALETTE["red"], "#8e44ad", "#16a085", "#d35400"]

color_idx = 0
for name, curve in aligned_curves.items():
    if name == BENCHMARK_NAME:
        ax.plot(curve.index, curve.values, label=name,
                color=PALETTE["benchmark"], linewidth=2.5,
                linestyle="--", zorder=10)
    else:
        ax.plot(curve.index, curve.values, label=name,
                linewidth=2.0, color=colors_cycle[color_idx % len(colors_cycle)])
        color_idx += 1

ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
ax.set_ylabel("Growth of €1")
ax.legend(loc="upper left", frameon=True, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("€%.2f"))

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(str(OUTPUT_DIR / "top_factors_b1_aligned.png"),
            dpi=200, bbox_inches="tight")
plt.close(fig)


# ---- Chart C: IR + IC summary ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Factor Signal Strength Summary",
             fontsize=16, fontweight="bold", color=PALETTE["darkblue"])

plot_data = comp.sort_values("Information Ratio", ascending=True).tail(15)

# IR
colors_ir = [PALETTE["green"] if v > 0 else PALETTE["red"]
             for v in plot_data["Information Ratio"]]
ax1.barh(plot_data["Factor"], plot_data["Information Ratio"],
         color=colors_ir, edgecolor="white", linewidth=0.8, height=0.65)
ax1.axvline(0, color="black", linewidth=0.8)
ax1.set_xlabel("Information Ratio")
ax1.set_title(f"B1 Information Ratio (vs {BENCHMARK_NAME})")
for i, (_, row) in enumerate(plot_data.iterrows()):
    val = row["Information Ratio"]
    if pd.notna(val):
        ax1.text(val + 0.02 * (1 if val >= 0 else -1), i, f"{val:.2f}",
                 va="center", fontsize=9)

# IC t-stat
colors_ic = [PALETTE["blue"] if v > 1.96 else
             PALETTE["green"] if v > 0 else PALETTE["red"]
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


print(f"\n\nResults saved to: {OUTPUT_DIR}/")
print(f"  all_factors_summary.csv         — comparison table")
print(f"  <factor_id>.png                 — individual 2x2 panels")
print(f"  top_factors_bar.png             — bucket bar chart (top {top_n})")
print(f"  top_factors_b1_aligned.png      — aligned B1 equity curves")
print(f"  factor_signal_strength.png      — IR + IC summary")
