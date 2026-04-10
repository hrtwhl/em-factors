"""
strategy_backtest.py — Long-only top-bucket strategies with daily equity curves
================================================================================

For each selected factor:
  - Each rebalancing date (monthly or quarterly): rank, pick B1
  - Hold equal-weighted between rebalancing dates
  - Apply transaction costs as one-day drag on rebalancing days
  - Compare to MSCI EM benchmark
  - All equity curves computed from DAILY price data
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Set, Optional

from factor_backtest import (
    load_daily_prices, portfolio_daily_returns, annualised_stats,
    align_to_common_start, apply_chart_style, PALETTE, BENCHMARK_NAME,
)

apply_chart_style()


# Configuration
DATA_DIR    = Path("backtest_data")
OUTPUT_DIR  = Path("strategy_results")
PRICE_FILE  = "EM_Indices_EUR.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

PORTFOLIO_VALUE = 60_000.0
COST_PER_TRADE  = 81.0
N_BUCKETS       = 5
MIN_COUNTRIES   = 10

STRATEGIES = [
    ("profit_margin",        "Profit Margin",                True),
    ("mom_1m",               "1M Momentum (Reversal)",       False),
    ("trailing_roa",         "Trailing ROA",                 True),
    ("earnings_revision_3m", "Earnings Revision (3M)",       True),
    ("risk_adj_mom_12m",     "Risk-Adjusted Momentum (12M)", True),
    ("cpi_3m_change",        "CPI 3M Change",                False),
    ("forward_roe",          "Forward ROE",                  True),
]


# =========================================================================
# Strategy result
# =========================================================================

@dataclass
class StrategyResult:
    factor_id: str
    factor_label: str
    rebal_freq: str
    higher_is_better: bool
    daily_returns_gross: pd.Series
    daily_returns_net: pd.Series
    holdings_history: list
    n_trades_per_rebal: dict
    benchmark_daily_returns: Optional[pd.Series]

    @property
    def equity_gross(self):
        return (1 + self.daily_returns_gross.fillna(0)).cumprod()

    @property
    def equity_net(self):
        return (1 + self.daily_returns_net.fillna(0)).cumprod()

    @property
    def equity_benchmark(self):
        if self.benchmark_daily_returns is None:
            return None
        bm = self.benchmark_daily_returns.reindex(self.daily_returns_gross.index)
        return (1 + bm.fillna(0)).cumprod()

    def summary(self):
        af = 252
        net = annualised_stats(self.daily_returns_net, af)
        gross = annualised_stats(self.daily_returns_gross, af)

        bm_stats = {}
        excess_data = {}
        if self.benchmark_daily_returns is not None:
            bm_aligned = self.benchmark_daily_returns.reindex(
                self.daily_returns_net.index)
            bm_stats = annualised_stats(bm_aligned, af)

            excess = (self.daily_returns_net - bm_aligned).dropna()
            if len(excess) > 0:
                ann_ex = (1 + excess.mean()) ** af - 1
                te = excess.std() * np.sqrt(af)
                excess_data = {
                    "Excess Ann Ret": ann_ex,
                    "Tracking Error": te,
                    "Information Ratio": ann_ex / te if te > 0 else np.nan,
                }

        total_trades = sum(self.n_trades_per_rebal.values())
        total_cost_eur = total_trades * COST_PER_TRADE
        years = len(self.daily_returns_net) / af

        return {
            "Factor": self.factor_label,
            "Rebal": self.rebal_freq,
            "Days": len(self.daily_returns_net),
            "Net Ann Ret": net.get("Ann Return", np.nan),
            "Net Ann Vol": net.get("Ann Vol", np.nan),
            "Net Sharpe": net.get("Sharpe", np.nan),
            "Net Max DD": net.get("Max DD", np.nan),
            "Gross Ann Ret": gross.get("Ann Return", np.nan),
            "Gross Sharpe": gross.get("Sharpe", np.nan),
            "BM Ann Ret": bm_stats.get("Ann Return", np.nan),
            **excess_data,
            "Total Trades": total_trades,
            "Total Cost EUR": total_cost_eur,
            "Cost Drag/yr": (total_cost_eur / PORTFOLIO_VALUE / years
                             if years > 0 else 0),
        }

    def holding_frequency(self):
        freq = Counter()
        n_periods = len(self.holdings_history)
        for _, _, holdings in self.holdings_history:
            for c in holdings:
                freq[c] += 1
        return freq, n_periods


def run_strategy(factor_id, factor_label, higher_is_better,
                 factor_df, daily_country_prices, benchmark_daily_returns,
                 rebal_freq="monthly"):
    """
    Build a long-only top-bucket strategy with daily equity curve.

    Steps:
      1. Filter factor dates to rebalancing schedule
         (monthly = all factor dates, quarterly = every 3rd)
      2. At each rebalancing factor date t:
         - Rank countries, identify B1 (top quintile)
         - Hold from next trading day after t until next rebalancing date
      3. Compute daily portfolio returns from daily prices
      4. Apply transaction costs as one-day drag on rebalancing days
    """
    daily_returns = daily_country_prices.pct_change(fill_method=None)
    factor_dates = sorted(factor_df["date"].unique())

    if rebal_freq == "monthly":
        rebal_dates = factor_dates
    elif rebal_freq == "quarterly":
        rebal_dates = factor_dates[::3]
    else:
        raise ValueError(f"Unknown freq: {rebal_freq}")

    holdings_history = []
    n_trades_per_rebal = {}
    current_holdings = set()

    for i, fdate in enumerate(rebal_dates):
        # Get factor values
        fslice = factor_df[factor_df["date"] == fdate][
            ["country", "factor_value"]
        ].dropna(subset=["factor_value"])

        # Filter to countries with daily price coverage
        valid_cols = set(daily_country_prices.columns)
        fslice = fslice[fslice["country"].isin(valid_cols)]

        # --- 1. DETERMINE DATES FIRST ---
        # We must figure out the start/end dates regardless of missing data
        next_days = daily_returns.index[daily_returns.index > fdate]
        if len(next_days) == 0:
            continue
        start_date = next_days[0]

        if i + 1 < len(rebal_dates):
            next_fdate = rebal_dates[i + 1]
            next_next = daily_returns.index[daily_returns.index > next_fdate]
            end_date = next_next[0] if len(next_next) > 0 else None
        else:
            end_date = None

        # --- 2. THE STALENESS FIX ---
        if len(fslice) < MIN_COUNTRIES:
            # If we lack data, but we already own stocks, just keep holding them!
            if current_holdings:
                holdings_history.append((start_date, end_date, current_holdings))
                n_trades_per_rebal[start_date] = 0 # 0 trades = €0 cost
            continue

        # --- 3. RANK AND ASSIGN BUCKETS ---
        fslice = fslice.sort_values(
            "factor_value", ascending=not higher_is_better
        ).reset_index(drop=True)
        n = len(fslice)
        bucket_size = n / N_BUCKETS
        fslice["bucket"] = [
            min(int(j // bucket_size) + 1, N_BUCKETS) for j in range(n)
        ]
        new_holdings = set(fslice[fslice["bucket"] == 1]["country"])

        if not new_holdings:
            # Also hold if bucket sorting fails for some reason
            if current_holdings:
                holdings_history.append((start_date, end_date, current_holdings))
                n_trades_per_rebal[start_date] = 0
            continue

        # --- 4. CALCULATE TRADES & COSTS ---
        if not current_holdings:
            n_trades = len(new_holdings)
        else:
            sells = current_holdings - new_holdings
            buys = new_holdings - current_holdings
            n_trades = len(sells) + len(buys)

        n_trades_per_rebal[start_date] = n_trades
        holdings_history.append((start_date, end_date, new_holdings))
        current_holdings = new_holdings

    if not holdings_history:
        raise RuntimeError(f"No valid rebalances for {factor_label}")

    # Daily portfolio returns
    gross_daily = portfolio_daily_returns(daily_returns, holdings_history)

    daily_costs = {
        d: (n_trades * COST_PER_TRADE) / PORTFOLIO_VALUE
        for d, n_trades in n_trades_per_rebal.items()
    }
    net_daily = portfolio_daily_returns(daily_returns, holdings_history, daily_costs)

    return StrategyResult(
        factor_id=factor_id,
        factor_label=factor_label,
        rebal_freq=rebal_freq,
        higher_is_better=higher_is_better,
        daily_returns_gross=gross_daily,
        daily_returns_net=net_daily,
        holdings_history=holdings_history,
        n_trades_per_rebal=n_trades_per_rebal,
        benchmark_daily_returns=benchmark_daily_returns,
    )


# =========================================================================
# Plots
# =========================================================================

def plot_strategy(result, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(result.factor_label, fontsize=18, fontweight="bold",
                 color=PALETTE["darkblue"], y=0.995)
    fig.text(0.5, 0.965,
             f"{result.rebal_freq.title()} rebalancing  ·  "
             f"Daily equity curve  ·  Min {MIN_COUNTRIES} countries  ·  "
             f"€{COST_PER_TRADE:.0f}/trade on €{PORTFOLIO_VALUE:,.0f}",
             ha="center", fontsize=11, color=PALETTE["grey"])

    eq_gross = result.equity_gross
    eq_net = result.equity_net
    eq_bm = result.equity_benchmark

    # 1. Cumulative returns
    ax = axes[0, 0]
    ax.plot(eq_gross.index, eq_gross, label="Gross",
            color=PALETTE["blue"], linewidth=2)
    ax.plot(eq_net.index, eq_net, label="Net of costs",
            color=PALETTE["darkblue"], linewidth=2, linestyle="--")
    if eq_bm is not None:
        ax.plot(eq_bm.index, eq_bm, label=BENCHMARK_NAME,
                color=PALETTE["benchmark"], linewidth=2, alpha=0.85,
                linestyle=":")
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Growth of €1")
    ax.legend(frameon=True)

    # 2. Rolling 12M excess (computed monthly)
    ax = axes[0, 1]
    if eq_bm is not None:
        net_m = result.daily_returns_net.resample("ME").apply(
            lambda x: (1 + x).prod() - 1)
        bm_aligned = result.benchmark_daily_returns.reindex(
            result.daily_returns_net.index)
        bm_m = bm_aligned.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        excess_m = (net_m - bm_m).dropna()
        if len(excess_m) > 12:
            rolling = excess_m.rolling(12).sum()
            ax.plot(rolling.index, rolling, color=PALETTE["blue"], linewidth=1.8)
            ax.fill_between(rolling.index, 0, rolling,
                            where=(rolling >= 0), alpha=0.18,
                            color=PALETTE["green"])
            ax.fill_between(rolling.index, 0, rolling,
                            where=(rolling < 0), alpha=0.18,
                            color=PALETTE["red"])
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Rolling 12M Excess Return  (Net vs {BENCHMARK_NAME})")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # 3. Drawdowns
    ax = axes[1, 0]
    dd_net = eq_net / eq_net.cummax() - 1
    ax.fill_between(dd_net.index, dd_net, 0, alpha=0.4,
                    color=PALETTE["blue"], label="Strategy (Net)")
    if eq_bm is not None:
        dd_bm = eq_bm / eq_bm.cummax() - 1
        ax.plot(dd_bm.index, dd_bm, color=PALETTE["benchmark"],
                linewidth=1.8, linestyle="--", alpha=0.85, label=BENCHMARK_NAME)
    ax.set_title("Drawdowns")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(frameon=True)

    # 4. Trades per rebalancing
    ax = axes[1, 1]
    if result.n_trades_per_rebal:
        dates = sorted(result.n_trades_per_rebal.keys())
        trades = [result.n_trades_per_rebal[d] for d in dates]
        ax.bar(dates, trades, color=PALETTE["orange"], alpha=0.75, width=15)
        ax.set_title("Trades per Rebalancing")
        ax.set_ylabel("# Trades")
        ax.set_ylim(0, max(trades) + 1 if trades else 1)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(all_results, save_path):
    """Side-by-side: monthly and quarterly, all aligned to common start."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Strategy Equity Curves — Net of Transaction Costs",
                 fontsize=16, fontweight="bold", color=PALETTE["darkblue"])

    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"],
              PALETTE["red"], "#8e44ad", "#16a085", "#d35400", "#27ae60"]

    for ax_idx, freq in enumerate(["monthly", "quarterly"]):
        ax = axes[ax_idx]
        freq_results = [r for r in all_results if r.rebal_freq == freq]
        if not freq_results:
            continue

        curves = {r.factor_label: r.equity_net for r in freq_results}

        if freq_results[0].benchmark_daily_returns is not None:
            bm = freq_results[0].benchmark_daily_returns
            curves[BENCHMARK_NAME] = (1 + bm.fillna(0)).cumprod()

        aligned = align_to_common_start(curves)

        if aligned:
            common_start = min(c.index[0] for c in aligned.values())
            ax.text(0.02, 0.98, f"From {common_start.date()}",
                    transform=ax.transAxes, fontsize=8,
                    color=PALETTE["grey"], va="top")

        color_idx = 0
        for name, curve in aligned.items():
            if name == BENCHMARK_NAME:
                ax.plot(curve.index, curve.values, label=name,
                        color=PALETTE["benchmark"], linewidth=2.5,
                        linestyle="--", zorder=10)
            else:
                ax.plot(curve.index, curve.values, label=name,
                        linewidth=1.8, color=colors[color_idx % len(colors)])
                color_idx += 1

        ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{freq.title()} Rebalancing", fontsize=13)
        ax.set_ylabel("Growth of €1")
        ax.legend(loc="upper left", fontsize=8, frameon=True)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("€%.2f"))

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_holding_frequency(result, save_path):
    """Generates and saves a bar chart of the B1 holding frequencies."""
    freq, n_periods = result.holding_frequency()
    if not freq:
        return

    sorted_freq = freq.most_common()
    countries = [x[0] for x in sorted_freq][::-1]
    counts = [x[1] for x in sorted_freq][::-1]
    percentages = [c / n_periods for c in counts]

    fig, ax = plt.subplots(figsize=(10, max(6, len(countries) * 0.3)))
    
    ax.barh(countries, percentages, color=PALETTE["darkblue"], alpha=0.75)
    ax.set_title(f"B1 Holdings Frequency: {result.factor_label} ({result.rebal_freq.title()})", 
                 fontsize=14, fontweight="bold", color=PALETTE["darkblue"])
    ax.set_xlabel(f"Frequency in Top Bucket (Total periods: {n_periods})")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Explicitly disable gridlines in case the global style applies them
    ax.grid(False)

    # Add text labels next to the bars (percentage only)
    for i, pct in enumerate(percentages):
        ax.text(pct + 0.01, i, f"{pct:.1%}", va='center', fontsize=9)

    # Expand x-axis slightly so the text doesn't get cut off
    ax.set_xlim(0, max(percentages) * 1.15 if percentages else 1.0)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 80)
    print("  STRATEGY BACKTEST v3 — Daily Equity Curves")
    print(f"  Portfolio: €{PORTFOLIO_VALUE:,.0f}  |  Cost: €{COST_PER_TRADE:.0f}/trade")
    print(f"  Buckets: {N_BUCKETS} → B1 = top {15 // N_BUCKETS} countries")
    print(f"  Min countries: {MIN_COUNTRIES}")
    print("=" * 80)

    print("\nLoading daily prices...")
    country_prices, benchmark_prices = load_daily_prices(PRICE_FILE)
    benchmark_returns = (benchmark_prices.pct_change()
                         if benchmark_prices is not None else None)
    print(f"  {country_prices.shape[1]} countries, "
          f"{country_prices.shape[0]} trading days")
    print(f"  Date range: {country_prices.index.min().date()} to "
          f"{country_prices.index.max().date()}")
    print(f"  Benchmark: {'OK' if benchmark_prices is not None else 'NOT FOUND'}\n")

    all_results = []

    for fid, label, higher in STRATEGIES:
        factor_file = DATA_DIR / f"factor_{fid}.csv"
        if not factor_file.exists():
            print(f"  SKIP {label}: file not found")
            continue

        factor_df = pd.read_csv(factor_file, parse_dates=["date"])

        for freq in ["monthly", "quarterly"]:
            print(f"  {label:35s} ({freq:9s}) ... ", end="", flush=True)

            try:
                result = run_strategy(
                    fid, label, higher,
                    factor_df, country_prices, benchmark_returns,
                    rebal_freq=freq,
                )
                all_results.append(result)

                s = result.summary()
                ir = s.get("Information Ratio", float("nan"))
                ir_str = f"{ir:.2f}" if pd.notna(ir) else "N/A"
                print(f"Net SR={s['Net Sharpe']:.2f}, "
                      f"Ret={s['Net Ann Ret']:+.1%}, "
                      f"IR={ir_str}, "
                      f"Trades={s['Total Trades']}, "
                      f"€{s['Total Cost EUR']:,.0f}")

                plot_strategy(result, OUTPUT_DIR / f"{fid}_{freq}.png")

                # Export holding frequency charts for monthly strategies
                if freq == "monthly":
                    plot_holding_frequency(result, OUTPUT_DIR / f"{fid}_monthly_holdings.png")

            except RuntimeError as e:
                print(f"FAILED: {e}")

    if not all_results:
        print("No strategies completed.")
        return

    # ---- Comparison tables -----------------------------------------------
    summaries = [r.summary() for r in all_results]
    comp = pd.DataFrame(summaries)

    print("\n\n" + "=" * 110)
    print("  STRATEGY COMPARISON")
    print("=" * 110)

    for freq in ["monthly", "quarterly"]:
        sub = comp[comp["Rebal"] == freq].sort_values(
            "Information Ratio", ascending=False)
        if len(sub) == 0:
            continue

        print(f"\n--- {freq.upper()} REBALANCING (sorted by Information Ratio) ---\n")

        cols = ["Factor", "Days", "Net Ann Ret", "Net Ann Vol",
                "Net Sharpe", "Net Max DD",
                "Gross Ann Ret", "BM Ann Ret",
                "Excess Ann Ret", "Information Ratio",
                "Total Trades", "Total Cost EUR", "Cost Drag/yr"]
        d = sub[[c for c in cols if c in sub.columns]].copy()

        for c in ["Net Ann Ret", "Net Ann Vol", "Net Max DD",
                   "Gross Ann Ret", "BM Ann Ret", "Excess Ann Ret",
                   "Cost Drag/yr"]:
            if c in d.columns:
                d[c] = d[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "N/A")
        for c in ["Net Sharpe", "Information Ratio"]:
            if c in d.columns:
                d[c] = d[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        d["Total Cost EUR"] = d["Total Cost EUR"].map(lambda x: f"€{x:,.0f}")

        print(d.to_string(index=False))

    # Monthly vs Quarterly side-by-side
    print("\n\n--- MONTHLY vs QUARTERLY ---\n")
    print(f"  {'Factor':<30s}  "
          f"{'--- Monthly ---':>40s}  |  {'--- Quarterly ---':>40s}")
    print(f"  {'':30s}  "
          f"{'Net SR':>7s} {'Net Ret':>8s} {'IR':>6s} {'Trades':>7s} {'Cost':>9s}"
          f"  |  "
          f"{'Net SR':>7s} {'Net Ret':>8s} {'IR':>6s} {'Trades':>7s} {'Cost':>9s}")
    print("  " + "-" * 110)

    for fid, label, _ in STRATEGIES:
        m = [r for r in all_results
             if r.factor_id == fid and r.rebal_freq == "monthly"]
        q = [r for r in all_results
             if r.factor_id == fid and r.rebal_freq == "quarterly"]
        if not m or not q:
            continue
        ms, qs = m[0].summary(), q[0].summary()

        def fmt(s):
            ir = s.get("Information Ratio", float("nan"))
            return (f"{s['Net Sharpe']:>7.2f} {s['Net Ann Ret']:>+7.1%} "
                    f"{ir:>6.2f} "
                    f"{s['Total Trades']:>6d} €{s['Total Cost EUR']:>7,.0f}")

        print(f"  {label:<30s}  {fmt(ms)}  |  {fmt(qs)}")

    # ---- Holdings frequency ----------------------------------------------
    print("\n\n--- B1 HOLDINGS FREQUENCY (All Monthly Strategies) ---\n")
    m_results = [r for r in all_results if r.rebal_freq == "monthly"]

    for r in m_results:
        freq, n_periods = r.holding_frequency()
        print(f"  {r.factor_label}:")
        for country, count in freq.most_common():
            bar = "█" * int(count / n_periods * 30)
            print(f"    {country:20s} {count:3d}/{n_periods} "
                  f"({count/n_periods:5.1%}) {bar}")
        print()

    # ---- Save ------------------------------------------------------------
    comp.to_csv(str(OUTPUT_DIR / "strategy_comparison.csv"), index=False)
    plot_comparison(all_results, OUTPUT_DIR / "all_strategies_comparison.png")

    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  strategy_comparison.csv         — comparison table")
    print(f"  all_strategies_comparison.png   — aligned equity curves")
    print(f"  <factor>_<freq>.png             — individual strategy panels")
    print(f"  <factor>_monthly_holdings.png   — holding frequency bar charts")


if __name__ == "__main__":
    main()