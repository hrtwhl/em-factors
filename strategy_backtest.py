"""
strategy_backtest.py — Long-only top-bucket strategy backtester (v2)
=====================================================================

Key improvements over v1:
  - min_countries raised to 10 (sorting <10 into quintiles is unreliable)
  - Full diagnostics: shows which countries are in B1 each period
  - Tracks "active" vs "holding" months separately
  - Compares results on SAME date set as quintile sort (apples-to-apples)
  - Detailed cost breakdown

Usage:
    python strategy_backtest.py
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Set


# =========================================================================
# Configuration
# =========================================================================

DATA_DIR   = Path("backtest_data")
OUTPUT_DIR = Path("strategy_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PORTFOLIO_VALUE  = 60_000.0
COST_PER_TRADE   = 81.0
N_BUCKETS        = 5
MIN_COUNTRIES    = 10   # raised from 5 — need at least 10 to form meaningful quintiles

PRICE_FILE = "EM_Indices_EUR.csv"

STRATEGIES = [
    ("trailing_pe",          "Trailing PE",                False),
    ("forward_pe",           "Forward PE",                 False),
    ("price_to_fcf",         "Price-to-Free-Cash-Flow",    False),
    ("price_to_cf",          "Price-to-Cash-Flow",         False),
    ("forward_roe",          "Forward ROE",                True),
    ("trailing_roa",         "Trailing ROA",               True),
    ("cpi_3m_change",        "CPI 3M Change",              False),
    ("sales_revision_3m",    "Sales Revision (3M)",        True),
]


# =========================================================================
# Data loading
# =========================================================================

def load_returns(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "returns.csv", parse_dates=["date"])
    return df.sort_values(["date", "country"]).reset_index(drop=True)


def load_factor(data_dir: Path, factor_id: str) -> pd.DataFrame:
    path = data_dir / f"factor_{factor_id}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["date", "country"]).reset_index(drop=True)


def load_benchmark(price_file: str) -> pd.Series:
    prices = pd.read_csv(price_file)
    prices.columns = [c.strip() for c in prices.columns]
    date_col = prices.columns[0]
    prices.rename(columns={date_col: "date"}, inplace=True)
    prices["date"] = pd.to_datetime(prices["date"])

    em_col = None
    for candidate in ["MIMUEMRN Index", "MIMUEMRN.Index"]:
        if candidate in prices.columns:
            em_col = candidate
            break

    if em_col is None:
        print("WARNING: MSCI EM benchmark not found.")
        return pd.Series(dtype=float)

    em = prices[["date", em_col]].dropna().copy()
    em.columns = ["date", "price"]
    em = em.sort_values("date")
    em["year_month"] = em["date"].dt.to_period("M")
    em_monthly = em.groupby("year_month").last().reset_index()
    em_monthly["return"] = em_monthly["price"] / em_monthly["price"].shift(1) - 1
    em_monthly = em_monthly.dropna(subset=["return"])
    em_monthly.index = em_monthly["date"]
    return em_monthly["return"].rename("MSCI_EM")


# =========================================================================
# Strategy engine
# =========================================================================

@dataclass
class PeriodDetail:
    """What happened in a single period."""
    date: pd.Timestamp
    factor_date: pd.Timestamp
    rebalanced: bool
    holdings: set
    n_countries_available: int
    n_trades: int
    cost_eur: float
    cost_pct: float
    gross_return: float
    net_return: float
    benchmark_return: float
    is_active: bool  # True if we had enough data to form buckets


@dataclass
class StrategyResult:
    factor_id: str
    factor_label: str
    rebal_freq: str
    higher_is_better: bool
    periods: List[PeriodDetail]
    benchmark_returns: Optional[pd.Series] = None

    @property
    def active_periods(self):
        return [p for p in self.periods if p.is_active]

    @property
    def holding_periods(self):
        return [p for p in self.periods if not p.is_active]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for p in self.periods:
            rows.append({
                "date": p.date,
                "gross_return": p.gross_return,
                "net_return": p.net_return,
                "benchmark": p.benchmark_return,
                "n_trades": p.n_trades,
                "cost_pct": p.cost_pct,
                "n_countries": p.n_countries_available,
                "rebalanced": p.rebalanced,
                "is_active": p.is_active,
                "holdings": "|".join(sorted(p.holdings)),
            })
        return pd.DataFrame(rows).set_index("date")

    def summary(self) -> dict:
        df = self.to_dataframe()
        af = 12

        def _stats(series, label):
            s = series.dropna()
            if len(s) == 0:
                return {}
            ann_ret = (1 + s.mean()) ** af - 1
            ann_vol = s.std() * np.sqrt(af)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            cum = (1 + s).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            hit = (s > 0).mean()
            total_ret = cum.iloc[-1] - 1
            return {
                f"{label} Ann. Ret": ann_ret,
                f"{label} Ann. Vol": ann_vol,
                f"{label} Sharpe": sharpe,
                f"{label} Max DD": max_dd,
                f"{label} Hit Rate": hit,
                f"{label} Total Ret": total_ret,
            }

        stats = {
            "Factor": self.factor_label,
            "Rebal": self.rebal_freq,
            "Total Months": len(df),
            "Active Months": df["is_active"].sum(),
            "Holding Months": (~df["is_active"]).sum(),
        }

        # All periods (the real strategy return)
        stats.update(_stats(df["net_return"], "Net"))
        stats.update(_stats(df["gross_return"], "Gross"))

        # Active periods only (comparable to quintile sort)
        active_df = df[df["is_active"]]
        stats.update(_stats(active_df["net_return"], "Active Net"))

        # Benchmark
        bm = df["benchmark"].dropna()
        stats.update(_stats(bm, "BM"))

        # Excess
        excess = (df["net_return"] - df["benchmark"]).dropna()
        if len(excess) > 0:
            stats["Excess Ann. Ret"] = (1 + excess.mean()) ** af - 1
            te = excess.std() * np.sqrt(af)
            stats["Tracking Error"] = te
            stats["Info Ratio"] = stats["Excess Ann. Ret"] / te if te > 0 else np.nan

        # Costs
        stats["Total Trades"] = int(df["n_trades"].sum())
        stats["Total Cost EUR"] = stats["Total Trades"] * COST_PER_TRADE
        stats["Ann. Cost Drag"] = df["cost_pct"].mean() * af

        return stats


def run_strategy(
    factor_id: str,
    factor_label: str,
    higher_is_better: bool,
    returns_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    benchmark: pd.Series,
    rebal_freq: str = "monthly",
) -> StrategyResult:
    """Run a single top-bucket strategy with full period-level detail."""

    factor_dates = sorted(factor_df["date"].unique())
    return_dates = sorted(returns_df["date"].unique())
    return_dates_arr = np.array(return_dates)

    # Build (factor_date, return_date) pairs
    pairs = []
    for fdate in factor_dates:
        candidates = return_dates_arr[return_dates_arr > fdate]
        if len(candidates) == 0:
            continue
        pairs.append((fdate, candidates[0]))

    if not pairs:
        raise RuntimeError(f"No valid date pairs for {factor_label}")

    current_holdings: Set[str] = set()
    period_details: List[PeriodDetail] = []
    rebal_counter = 0

    for i, (fdate, ret_date) in enumerate(pairs):
        # Rebalance decision
        do_rebalance = False
        if rebal_freq == "monthly":
            do_rebalance = True
        elif rebal_freq == "quarterly":
            if i == 0 or rebal_counter >= 2:
                do_rebalance = True
                rebal_counter = 0
            else:
                rebal_counter += 1

        # Get available data
        fslice = factor_df[factor_df["date"] == fdate][
            ["country", "factor_value"]
        ].dropna(subset=["factor_value"])

        rslice = returns_df[returns_df["date"] == ret_date][
            ["country", "return"]
        ].dropna(subset=["return"])

        merged = fslice.merge(rslice, on="country", how="inner")
        n_available = len(merged)

        # Get benchmark return for this date
        bm_ret = np.nan
        if benchmark is not None and len(benchmark) > 0:
            bm_match = benchmark.reindex([ret_date])
            if len(bm_match) > 0 and not bm_match.isna().all():
                bm_ret = bm_match.iloc[0]

        # Can we form meaningful buckets?
        is_active = (n_available >= MIN_COUNTRIES) and do_rebalance

        if is_active:
            # Rank and pick top bucket
            merged = merged.sort_values(
                "factor_value", ascending=not higher_is_better
            ).reset_index(drop=True)

            n = len(merged)
            bucket_size = n / N_BUCKETS
            merged["bucket"] = [
                min(int(j // bucket_size) + 1, N_BUCKETS) for j in range(n)
            ]
            new_holdings = set(merged[merged["bucket"] == 1]["country"])

        else:
            # Hold existing portfolio
            new_holdings = current_holdings.copy()

        # Skip if we have no holdings at all (very start, before first valid date)
        if not new_holdings and not current_holdings:
            continue

        # If new_holdings is empty (edge case), keep current
        if not new_holdings:
            new_holdings = current_holdings.copy()

        # Compute trades
        if not current_holdings:
            buys = new_holdings
            sells = set()
        else:
            sells = current_holdings - new_holdings
            buys = new_holdings - current_holdings

        n_trades = len(sells) + len(buys)
        cost_eur = n_trades * COST_PER_TRADE
        cost_pct = cost_eur / PORTFOLIO_VALUE

        # Portfolio return
        ret_data = returns_df[returns_df["date"] == ret_date]
        held_rets = ret_data[ret_data["country"].isin(new_holdings)]["return"]

        if len(held_rets) == 0:
            # Fallback to current holdings
            held_rets = ret_data[ret_data["country"].isin(current_holdings)]["return"]
            if len(held_rets) == 0:
                continue
            new_holdings = current_holdings.copy()
            n_trades = 0
            cost_eur = 0
            cost_pct = 0

        port_ret = held_rets.mean()

        period_details.append(PeriodDetail(
            date=ret_date,
            factor_date=fdate,
            rebalanced=is_active or (not current_holdings),
            holdings=new_holdings.copy(),
            n_countries_available=n_available,
            n_trades=n_trades,
            cost_eur=cost_eur,
            cost_pct=cost_pct,
            gross_return=port_ret,
            net_return=port_ret - cost_pct,
            benchmark_return=bm_ret,
            is_active=is_active or (not current_holdings),
        ))

        current_holdings = new_holdings.copy()

    return StrategyResult(
        factor_id=factor_id,
        factor_label=factor_label,
        rebal_freq=rebal_freq,
        higher_is_better=higher_is_better,
        periods=period_details,
        benchmark_returns=benchmark,
    )


# =========================================================================
# Diagnostics
# =========================================================================

def print_diagnostics(result: StrategyResult):
    """Print detailed diagnostics for a single strategy."""
    df = result.to_dataframe()
    n_active = df["is_active"].sum()
    n_hold = (~df["is_active"]).sum()

    print(f"\n    Periods: {len(df)} total "
          f"({n_active} active rebalancing, {n_hold} holding)")
    print(f"    Countries available per active period: "
          f"min={df.loc[df['is_active'], 'n_countries'].min():.0f}, "
          f"median={df.loc[df['is_active'], 'n_countries'].median():.0f}, "
          f"max={df.loc[df['is_active'], 'n_countries'].max():.0f}")

    # Holding composition frequency
    all_holdings = []
    for p in result.periods:
        for c in p.holdings:
            all_holdings.append(c)
    if all_holdings:
        from collections import Counter
        freq = Counter(all_holdings)
        total = len(result.periods)
        print(f"    B1 composition (% of months held):")
        for country, count in freq.most_common():
            print(f"      {country:20s}  {count:4d}/{total} = {count/total:.0%}")


# =========================================================================
# Plotting
# =========================================================================

def plot_strategy(result: StrategyResult, save_path: Path):
    df = result.to_dataframe()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    title = (f"Strategy: {result.factor_label} — "
             f"{result.rebal_freq.title()} Rebalancing  "
             f"(min {MIN_COUNTRIES} countries)")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # 1. Cumulative returns
    ax = axes[0, 0]
    cum_gross = (1 + df["gross_return"]).cumprod()
    cum_net = (1 + df["net_return"]).cumprod()
    ax.plot(cum_gross.index, cum_gross, label="Gross", color="steelblue",
            linewidth=1.8)
    ax.plot(cum_net.index, cum_net, label="Net of costs", color="darkblue",
            linewidth=1.8, linestyle="--")
    bm = df["benchmark"].dropna()
    if len(bm) > 0:
        cum_bm = (1 + bm).cumprod()
        ax.plot(cum_bm.index, cum_bm, label="MSCI EM", color="grey",
                linewidth=1.5, alpha=0.7)
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Growth of €1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Rolling 12M excess
    ax = axes[0, 1]
    excess = (df["net_return"] - df["benchmark"]).dropna()
    if len(excess) > 12:
        rolling_ex = excess.rolling(12).mean() * 12
        ax.plot(rolling_ex.index, rolling_ex, color="steelblue", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Rolling 12M Annualised Excess (Net vs MSCI EM)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    # 3. Drawdowns
    ax = axes[1, 0]
    cum_net2 = (1 + df["net_return"]).cumprod()
    dd = cum_net2 / cum_net2.cummax() - 1
    ax.fill_between(dd.index, dd, 0, alpha=0.4, color="steelblue",
                    label="Strategy Net")
    if len(bm) > 0:
        cum_bm2 = (1 + bm).cumprod()
        dd_bm = cum_bm2 / cum_bm2.cummax() - 1
        ax.plot(dd_bm.index, dd_bm, color="grey", linewidth=1, alpha=0.7,
                label="MSCI EM")
    ax.set_title("Drawdowns")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Countries available + trades
    ax = axes[1, 1]
    ax.bar(df.index, df["n_countries"], color="lightblue", alpha=0.6,
           width=25, label="Countries available")
    ax.axhline(MIN_COUNTRIES, color="red", linewidth=1, linestyle="--",
               label=f"Min threshold ({MIN_COUNTRIES})")
    ax2 = ax.twinx()
    trades = df["n_trades"]
    ax2.bar(df.index[trades > 0], trades[trades > 0], color="salmon",
            alpha=0.6, width=15, label="Trades")
    ax.set_title("Data Coverage & Trades")
    ax.set_ylabel("Countries with data")
    ax2.set_ylabel("# Trades")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(all_results: list, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"All Strategies — Net of Costs (min {MIN_COUNTRIES} countries)",
        fontsize=14, fontweight="bold",
    )

    for freq_idx, freq in enumerate(["monthly", "quarterly"]):
        ax = axes[freq_idx]
        results_freq = [r for r in all_results if r.rebal_freq == freq]
        if not results_freq:
            continue

        bm_plotted = False
        for r in results_freq:
            df = r.to_dataframe()
            if not bm_plotted:
                bm = df["benchmark"].dropna()
                if len(bm) > 0:
                    cum_bm = (1 + bm).cumprod()
                    ax.plot(cum_bm.index, cum_bm, color="black", linewidth=2,
                            alpha=0.5, linestyle="--", label="MSCI EM", zorder=0)
                    bm_plotted = True

        colors = plt.cm.tab10(np.linspace(0, 1, len(results_freq)))
        for i, r in enumerate(results_freq):
            df = r.to_dataframe()
            cum = (1 + df["net_return"]).cumprod()
            ax.plot(cum.index, cum.values, label=r.factor_label,
                    linewidth=1.5, color=colors[i])

        ax.set_title(f"{freq.title()} Rebalancing", fontsize=13)
        ax.set_ylabel("Growth of €1")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 80)
    print("  STRATEGY BACKTEST v2: Long-Only Top-Bucket Strategies")
    print(f"  Portfolio: €{PORTFOLIO_VALUE:,.0f}  |  Cost: €{COST_PER_TRADE:.0f}/trade")
    print(f"  Buckets: {N_BUCKETS}  |  Min countries: {MIN_COUNTRIES}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    returns_df = load_returns(DATA_DIR)
    print(f"  Returns: {len(returns_df)} rows, "
          f"{returns_df['country'].nunique()} countries, "
          f"{returns_df['date'].nunique()} dates")

    benchmark = load_benchmark(PRICE_FILE)
    print(f"  Benchmark: {len(benchmark)} monthly observations")

    # Run all strategies
    all_results = []

    for factor_id, factor_label, higher_is_better in STRATEGIES:
        factor_file = DATA_DIR / f"factor_{factor_id}.csv"
        if not factor_file.exists():
            print(f"\nSKIPPING {factor_label}: file not found")
            continue

        factor_df = load_factor(DATA_DIR, factor_id)

        for freq in ["monthly", "quarterly"]:
            print(f"\n{'='*70}")
            print(f"  {factor_label} — {freq.title()}")
            print(f"{'='*70}")

            try:
                result = run_strategy(
                    factor_id=factor_id,
                    factor_label=factor_label,
                    higher_is_better=higher_is_better,
                    returns_df=returns_df,
                    factor_df=factor_df,
                    benchmark=benchmark,
                    rebal_freq=freq,
                )
                all_results.append(result)

                # Diagnostics
                print_diagnostics(result)

                # Summary stats
                s = result.summary()
                print(f"\n    --- Performance ---")
                print(f"    Net:   Ann Ret={s['Net Ann. Ret']:+.2%}, "
                      f"Vol={s['Net Ann. Vol']:.2%}, "
                      f"Sharpe={s['Net Sharpe']:.2f}, "
                      f"MaxDD={s['Net Max DD']:+.2%}")
                print(f"    Gross: Ann Ret={s['Gross Ann. Ret']:+.2%}, "
                      f"Sharpe={s['Gross Sharpe']:.2f}")
                if 'BM Ann. Ret' in s:
                    print(f"    MSCI EM: Ann Ret={s['BM Ann. Ret']:+.2%}")
                    print(f"    Excess={s.get('Excess Ann. Ret', np.nan):+.2%}, "
                          f"IR={s.get('Info Ratio', np.nan):.2f}")
                print(f"    Trades: {s['Total Trades']}, "
                      f"Cost €{s['Total Cost EUR']:,.0f} "
                      f"({s['Ann. Cost Drag']:.2%}/yr drag)")

                # Plot
                plot_path = OUTPUT_DIR / f"{factor_id}_{freq}.png"
                plot_strategy(result, plot_path)

            except RuntimeError as e:
                print(f"  FAILED: {e}")

    # ---- Comparison tables -----------------------------------------------
    if not all_results:
        print("No strategies completed.")
        return

    summaries = [r.summary() for r in all_results]
    comp = pd.DataFrame(summaries)

    print("\n\n" + "=" * 110)
    print("  STRATEGY COMPARISON")
    print("=" * 110)

    for freq in ["monthly", "quarterly"]:
        sub = comp[comp["Rebal"] == freq].sort_values("Net Sharpe", ascending=False)
        if len(sub) == 0:
            continue

        print(f"\n--- {freq.upper()} REBALANCING ---\n")

        display = sub[[
            "Factor", "Total Months", "Active Months",
            "Net Ann. Ret", "Net Ann. Vol", "Net Sharpe", "Net Max DD",
            "Gross Ann. Ret", "Gross Sharpe",
            "BM Ann. Ret", "Excess Ann. Ret", "Info Ratio",
            "Total Trades", "Total Cost EUR", "Ann. Cost Drag",
        ]].copy()

        for c in ["Net Ann. Ret", "Net Ann. Vol", "Net Max DD",
                   "Gross Ann. Ret", "BM Ann. Ret", "Excess Ann. Ret",
                   "Ann. Cost Drag"]:
            if c in display.columns:
                display[c] = display[c].map(
                    lambda x: f"{x:+.2%}" if pd.notna(x) else "N/A")
        display["Net Sharpe"] = display["Net Sharpe"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display["Gross Sharpe"] = display["Gross Sharpe"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display["Info Ratio"] = display["Info Ratio"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display["Total Cost EUR"] = display["Total Cost EUR"].map(
            lambda x: f"€{x:,.0f}")

        display.columns = [
            "Factor", "Months", "Active",
            "Net Ret", "Net Vol", "Net SR", "Max DD",
            "Gross Ret", "Gross SR",
            "BM Ret", "Excess", "IR",
            "Trades", "Cost €", "Cost/yr",
        ]
        print(display.to_string(index=False))

    # Monthly vs Quarterly
    print("\n\n--- MONTHLY vs QUARTERLY ---\n")
    header = (f"  {'Factor':<30s}  "
              f"{'--- Monthly ---':>46s}  |  "
              f"{'--- Quarterly ---':>46s}")
    print(header)
    print(f"  {'':30s}  "
          f"{'Net SR':>7s} {'Net Ret':>8s} {'Gross Ret':>9s} "
          f"{'Trades':>6s} {'Cost':>9s} {'Drag/yr':>8s}"
          f"  |  "
          f"{'Net SR':>7s} {'Net Ret':>8s} {'Gross Ret':>9s} "
          f"{'Trades':>6s} {'Cost':>9s} {'Drag/yr':>8s}")
    print("  " + "-" * 130)

    for fid, label, _ in STRATEGIES:
        m = [r for r in all_results if r.factor_id == fid and r.rebal_freq == "monthly"]
        q = [r for r in all_results if r.factor_id == fid and r.rebal_freq == "quarterly"]
        if not m or not q:
            continue
        ms, qs = m[0].summary(), q[0].summary()

        print(f"  {label:<30s}  "
              f"{ms['Net Sharpe']:>7.2f} {ms['Net Ann. Ret']:>+7.1%} "
              f"{ms['Gross Ann. Ret']:>+8.1%} "
              f"{ms['Total Trades']:>5d} €{ms['Total Cost EUR']:>7,.0f} "
              f"{ms['Ann. Cost Drag']:>+7.2%}"
              f"  |  "
              f"{qs['Net Sharpe']:>7.2f} {qs['Net Ann. Ret']:>+7.1%} "
              f"{qs['Gross Ann. Ret']:>+8.1%} "
              f"{qs['Total Trades']:>5d} €{qs['Total Cost EUR']:>7,.0f} "
              f"{qs['Ann. Cost Drag']:>+7.2%}")

    # Save
    comp.to_csv(str(OUTPUT_DIR / "strategy_comparison.csv"), index=False)
    plot_comparison(all_results, OUTPUT_DIR / "all_strategies_comparison.png")

    for r in all_results:
        r.to_dataframe().to_csv(
            str(OUTPUT_DIR / f"returns_{r.factor_id}_{r.rebal_freq}.csv"))

    # Save holdings detail for inspection
    for r in all_results:
        df = r.to_dataframe()
        detail_path = OUTPUT_DIR / f"holdings_{r.factor_id}_{r.rebal_freq}.csv"
        df[["holdings", "n_countries", "rebalanced", "is_active",
            "n_trades", "gross_return", "net_return", "benchmark"]].to_csv(
            str(detail_path))

    print(f"\n\nResults saved to: {OUTPUT_DIR}/")
    print(f"  strategy_comparison.csv        — comparison table")
    print(f"  all_strategies_comparison.png  — equity curves")
    print(f"  holdings_<factor>_<freq>.csv   — period-by-period holdings detail")


if __name__ == "__main__":
    main()
