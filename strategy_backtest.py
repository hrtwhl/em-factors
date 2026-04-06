"""
strategy_backtest.py — Long-only top-bucket strategy backtester (v2)
=====================================================================

For each factor: rank countries, buy B1 equal-weighted, track turnover,
apply transaction costs (€81/trade on €60k portfolio), benchmark vs MSCI EM.
Tests monthly and quarterly rebalancing.

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
from collections import Counter
from dataclasses import dataclass
from typing import Optional, List, Set
from factor_backtest import apply_chart_style, PALETTE, BUCKET_COLORS

apply_chart_style()

# =========================================================================
# Configuration
# =========================================================================

DATA_DIR   = Path("backtest_data")
OUTPUT_DIR = Path("strategy_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PORTFOLIO_VALUE  = 60_000.0
COST_PER_TRADE   = 81.0
N_BUCKETS        = 5
MIN_COUNTRIES    = 10

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

def load_returns(data_dir):
    df = pd.read_csv(data_dir / "returns.csv", parse_dates=["date"])
    return df.sort_values(["date", "country"]).reset_index(drop=True)

def load_factor(data_dir, factor_id):
    df = pd.read_csv(data_dir / f"factor_{factor_id}.csv", parse_dates=["date"])
    return df.sort_values(["date", "country"]).reset_index(drop=True)

def load_benchmark(price_file):
    prices = pd.read_csv(price_file)
    prices.columns = [c.strip() for c in prices.columns]
    prices.rename(columns={prices.columns[0]: "date"}, inplace=True)
    prices["date"] = pd.to_datetime(prices["date"])

    em_col = None
    for c in ["MIMUEMRN Index", "MIMUEMRN.Index"]:
        if c in prices.columns:
            em_col = c
            break
    if em_col is None:
        print("WARNING: MSCI EM benchmark not found.")
        return pd.Series(dtype=float)

    em = prices[["date", em_col]].dropna().copy()
    em.columns = ["date", "price"]
    em = em.sort_values("date")
    em["ym"] = em["date"].dt.to_period("M")
    em = em.groupby("ym").last().reset_index()
    em["return"] = em["price"] / em["price"].shift(1) - 1
    em = em.dropna(subset=["return"])
    em.index = em["date"]
    return em["return"].rename("MSCI_EM")


# =========================================================================
# Strategy engine
# =========================================================================

@dataclass
class PeriodDetail:
    date: pd.Timestamp
    holdings: set
    n_countries_available: int
    n_trades: int
    cost_pct: float
    gross_return: float
    net_return: float
    benchmark_return: float
    rebalanced: bool


@dataclass
class StrategyResult:
    factor_id: str
    factor_label: str
    rebal_freq: str
    higher_is_better: bool
    periods: List[PeriodDetail]

    def to_dataframe(self):
        rows = [{
            "date": p.date, "gross_return": p.gross_return,
            "net_return": p.net_return, "benchmark": p.benchmark_return,
            "n_trades": p.n_trades, "cost_pct": p.cost_pct,
            "n_countries": p.n_countries_available,
            "rebalanced": p.rebalanced,
            "holdings": "|".join(sorted(p.holdings)),
        } for p in self.periods]
        return pd.DataFrame(rows).set_index("date")

    def summary(self):
        df = self.to_dataframe()
        af = 12
        def _s(series, lbl):
            s = series.dropna()
            if len(s) == 0: return {}
            ann_ret = (1 + s.mean()) ** af - 1
            ann_vol = s.std() * np.sqrt(af)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            cum = (1 + s).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            return {f"{lbl} Ann Ret": ann_ret, f"{lbl} Vol": ann_vol,
                    f"{lbl} Sharpe": sharpe, f"{lbl} Max DD": max_dd,
                    f"{lbl} Hit Rate": (s > 0).mean(),
                    f"{lbl} Total Ret": cum.iloc[-1] - 1}

        stats = {"Factor": self.factor_label, "Rebal": self.rebal_freq,
                 "Months": len(df)}
        stats.update(_s(df["net_return"], "Net"))
        stats.update(_s(df["gross_return"], "Gross"))
        stats.update(_s(df["benchmark"].dropna(), "BM"))

        excess = (df["net_return"] - df["benchmark"]).dropna()
        if len(excess) > 0:
            stats["Excess Ann Ret"] = (1 + excess.mean()) ** af - 1
            te = excess.std() * np.sqrt(af)
            stats["Tracking Error"] = te
            stats["Info Ratio"] = stats["Excess Ann Ret"] / te if te > 0 else np.nan

        stats["Total Trades"] = int(df["n_trades"].sum())
        stats["Total Cost EUR"] = stats["Total Trades"] * COST_PER_TRADE
        stats["Ann Cost Drag"] = df["cost_pct"].mean() * af
        return stats

    def holding_frequency(self):
        freq = Counter()
        for p in self.periods:
            for c in p.holdings:
                freq[c] += 1
        return freq


def run_strategy(factor_id, factor_label, higher_is_better,
                 returns_df, factor_df, benchmark, rebal_freq="monthly"):
    """
    Each rebalancing date:
      1. Observe factor at date t
      2. Find return date t+1 (strictly after t)
      3. If ≥ MIN_COUNTRIES countries have both factor[t] and return[t+1]:
         rank, pick B1, compute trades vs previous holdings
      4. Otherwise: hold existing portfolio (no trades)
      5. Record the return over month t+1 for whatever we hold
    """
    factor_dates = sorted(factor_df["date"].unique())
    return_dates_arr = np.array(sorted(returns_df["date"].unique()))

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
        do_rebalance = (rebal_freq == "monthly")
        if rebal_freq == "quarterly":
            if i == 0 or rebal_counter >= 2:
                do_rebalance = True
                rebal_counter = 0
            else:
                rebal_counter += 1

        # Get data
        fslice = factor_df[factor_df["date"] == fdate][
            ["country", "factor_value"]].dropna(subset=["factor_value"])
        rslice = returns_df[returns_df["date"] == ret_date][
            ["country", "return"]].dropna(subset=["return"])
        merged = fslice.merge(rslice, on="country", how="inner")
        n_available = len(merged)

        # Benchmark
        bm_ret = np.nan
        if benchmark is not None and len(benchmark) > 0:
            bm_match = benchmark.reindex([ret_date])
            if len(bm_match) > 0 and not bm_match.isna().all():
                bm_ret = bm_match.iloc[0]

        # Attempt rebalancing
        if do_rebalance and n_available >= MIN_COUNTRIES:
            merged = merged.sort_values(
                "factor_value", ascending=not higher_is_better
            ).reset_index(drop=True)
            n = len(merged)
            bucket_size = n / N_BUCKETS
            merged["bucket"] = [
                min(int(j // bucket_size) + 1, N_BUCKETS) for j in range(n)]
            new_holdings = set(merged[merged["bucket"] == 1]["country"])
            rebalanced = True
        else:
            new_holdings = current_holdings.copy()
            rebalanced = False

        # Need holdings to proceed
        if not new_holdings and not current_holdings:
            continue
        if not new_holdings:
            new_holdings = current_holdings.copy()

        # Trades
        if not current_holdings:
            n_trades = len(new_holdings)
        else:
            sells = current_holdings - new_holdings
            buys = new_holdings - current_holdings
            n_trades = len(sells) + len(buys)

        cost_pct = (n_trades * COST_PER_TRADE) / PORTFOLIO_VALUE

        # Return
        ret_data = returns_df[returns_df["date"] == ret_date]
        held_rets = ret_data[ret_data["country"].isin(new_holdings)]["return"]
        if len(held_rets) == 0:
            held_rets = ret_data[ret_data["country"].isin(current_holdings)]["return"]
            if len(held_rets) == 0:
                continue
            new_holdings = current_holdings.copy()
            n_trades = 0
            cost_pct = 0

        port_ret = held_rets.mean()

        period_details.append(PeriodDetail(
            date=ret_date, holdings=new_holdings.copy(),
            n_countries_available=n_available, n_trades=n_trades,
            cost_pct=cost_pct, gross_return=port_ret,
            net_return=port_ret - cost_pct,
            benchmark_return=bm_ret, rebalanced=rebalanced))

        current_holdings = new_holdings.copy()

    return StrategyResult(
        factor_id=factor_id, factor_label=factor_label,
        rebal_freq=rebal_freq, higher_is_better=higher_is_better,
        periods=period_details)


# =========================================================================
# Plotting
# =========================================================================

def plot_strategy(result, save_path):
    df = result.to_dataframe()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"{result.factor_label}",
                 fontsize=18, fontweight="bold", color=PALETTE["darkblue"], y=0.995)
    fig.text(0.5, 0.965,
             f"{result.rebal_freq.title()} rebalancing  ·  "
             f"Min {MIN_COUNTRIES} countries  ·  "
             f"€{COST_PER_TRADE:.0f}/trade on €{PORTFOLIO_VALUE:,.0f}",
             ha="center", fontsize=11, color=PALETTE["grey"])

    # 1. Cumulative returns
    ax = axes[0, 0]
    cum_g = (1 + df["gross_return"]).cumprod()
    cum_n = (1 + df["net_return"]).cumprod()
    ax.plot(cum_g.index, cum_g, label="Gross", color=PALETTE["blue"], linewidth=2)
    ax.plot(cum_n.index, cum_n, label="Net of costs", color=PALETTE["darkblue"],
            linewidth=2, linestyle="--")
    bm = df["benchmark"].dropna()
    if len(bm) > 0:
        cum_bm = (1 + bm).cumprod()
        ax.plot(cum_bm.index, cum_bm, label="MSCI EM", color=PALETTE["grey"],
                linewidth=1.8, alpha=0.7)
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Growth of €1")
    ax.legend(frameon=True)

    # 2. Rolling 12M excess
    ax = axes[0, 1]
    excess = (df["net_return"] - df["benchmark"]).dropna()
    if len(excess) > 12:
        rolling = excess.rolling(12).mean() * 12
        ax.plot(rolling.index, rolling, color=PALETTE["blue"], linewidth=1.8)
        ax.fill_between(rolling.index, 0, rolling,
                        where=rolling >= 0, alpha=0.15, color=PALETTE["green"])
        ax.fill_between(rolling.index, 0, rolling,
                        where=rolling < 0, alpha=0.15, color=PALETTE["red"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Rolling 12M Excess Return  (Net vs MSCI EM)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # 3. Drawdowns
    ax = axes[1, 0]
    cum_n2 = (1 + df["net_return"]).cumprod()
    dd = cum_n2 / cum_n2.cummax() - 1
    ax.fill_between(dd.index, dd, 0, alpha=0.4, color=PALETTE["blue"],
                    label="Strategy (Net)")
    if len(bm) > 0:
        cum_bm2 = (1 + bm).cumprod()
        dd_bm = cum_bm2 / cum_bm2.cummax() - 1
        ax.plot(dd_bm.index, dd_bm, color=PALETTE["grey"], linewidth=1.2,
                alpha=0.7, label="MSCI EM")
    ax.set_title("Drawdowns")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(frameon=True)

    # 4. Coverage & trades
    ax = axes[1, 1]
    ax.bar(df.index, df["n_countries"], color=PALETTE["blue"], alpha=0.25,
           width=25, label="Countries available")
    ax.axhline(MIN_COUNTRIES, color=PALETTE["red"], linewidth=1.2,
               linestyle="--", label=f"Min threshold ({MIN_COUNTRIES})")
    ax2 = ax.twinx()
    t = df["n_trades"]
    ax2.bar(df.index[t > 0], t[t > 0], color=PALETTE["orange"],
            alpha=0.7, width=15, label="Trades")
    ax.set_title("Data Coverage & Trades")
    ax.set_ylabel("Countries")
    ax2.set_ylabel("Trades")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax2.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(all_results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Strategy Equity Curves — Net of Transaction Costs",
                 fontsize=16, fontweight="bold", color=PALETTE["darkblue"])

    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"],
              PALETTE["red"], "#8e44ad", "#16a085", "#2c3e50", "#d35400"]

    for fi, freq in enumerate(["monthly", "quarterly"]):
        ax = axes[fi]
        freq_results = [r for r in all_results if r.rebal_freq == freq]
        if not freq_results:
            continue

        bm_done = False
        for r in freq_results:
            df = r.to_dataframe()
            if not bm_done:
                bm = df["benchmark"].dropna()
                if len(bm) > 0:
                    ax.plot((1 + bm).cumprod().index, (1 + bm).cumprod().values,
                            color="black", linewidth=2, alpha=0.4,
                            linestyle="--", label="MSCI EM", zorder=0)
                    bm_done = True

        for i, r in enumerate(freq_results):
            df = r.to_dataframe()
            cum = (1 + df["net_return"]).cumprod()
            ax.plot(cum.index, cum.values, label=r.factor_label,
                    linewidth=1.8, color=colors[i % len(colors)])

        ax.set_title(f"{freq.title()} Rebalancing", fontsize=13)
        ax.set_ylabel("Growth of €1")
        ax.legend(fontsize=8, loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 80)
    print("  STRATEGY BACKTEST v2")
    print(f"  Portfolio: €{PORTFOLIO_VALUE:,.0f}  |  Cost: €{COST_PER_TRADE:.0f}/trade")
    print(f"  Buckets: {N_BUCKETS} → B1 = {15 // N_BUCKETS} countries  |  "
          f"Min countries: {MIN_COUNTRIES}")
    print("=" * 80)

    returns_df = load_returns(DATA_DIR)
    benchmark = load_benchmark(PRICE_FILE)
    print(f"\n  Returns: {returns_df['country'].nunique()} countries, "
          f"{returns_df['date'].nunique()} dates")
    print(f"  Benchmark: {len(benchmark)} months\n")

    all_results = []

    for fid, label, higher in STRATEGIES:
        if not (DATA_DIR / f"factor_{fid}.csv").exists():
            print(f"  SKIP {label}: file not found")
            continue

        factor_df = load_factor(DATA_DIR, fid)

        for freq in ["monthly", "quarterly"]:
            print(f"  {label:35s} ({freq:9s}) ... ", end="", flush=True)

            try:
                result = run_strategy(fid, label, higher,
                                      returns_df, factor_df, benchmark, freq)
                all_results.append(result)

                s = result.summary()
                print(f"Net Sharpe={s['Net Sharpe']:.2f}, "
                      f"Ret={s['Net Ann Ret']:+.1%}, "
                      f"Trades={s['Total Trades']}, "
                      f"Cost €{s['Total Cost EUR']:,.0f}")

                plot_strategy(result, OUTPUT_DIR / f"{fid}_{freq}.png")

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
        sub = comp[comp["Rebal"] == freq].sort_values("Net Sharpe", ascending=False)
        if len(sub) == 0:
            continue

        print(f"\n--- {freq.upper()} REBALANCING ---\n")

        d = sub[["Factor", "Months",
                 "Net Ann Ret", "Net Vol", "Net Sharpe", "Net Max DD",
                 "Gross Ann Ret", "Gross Sharpe",
                 "BM Ann Ret", "Excess Ann Ret", "Info Ratio",
                 "Total Trades", "Total Cost EUR", "Ann Cost Drag"]].copy()

        for c in ["Net Ann Ret", "Net Vol", "Net Max DD",
                   "Gross Ann Ret", "BM Ann Ret", "Excess Ann Ret", "Ann Cost Drag"]:
            if c in d.columns:
                d[c] = d[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "N/A")
        d["Net Sharpe"] = d["Net Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        d["Gross Sharpe"] = d["Gross Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        d["Info Ratio"] = d["Info Ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        d["Total Cost EUR"] = d["Total Cost EUR"].map(lambda x: f"€{x:,.0f}")

        d.columns = ["Factor", "Months", "Net Ret", "Net Vol", "Net SR", "Max DD",
                      "Gross Ret", "Grs SR", "BM Ret", "Excess", "IR",
                      "Trades", "Cost €", "Cost/yr"]
        print(d.to_string(index=False))

    # Monthly vs Quarterly
    print("\n\n--- MONTHLY vs QUARTERLY ---\n")
    print(f"  {'Factor':<30s}  "
          f"{'--- Monthly ---':>44s}  |  {'--- Quarterly ---':>44s}")
    print(f"  {'':30s}  "
          f"{'Net SR':>7s} {'Net Ret':>8s} {'Grs Ret':>8s} "
          f"{'Trades':>6s} {'Cost':>8s} {'Drag/yr':>8s}"
          f"  |  "
          f"{'Net SR':>7s} {'Net Ret':>8s} {'Grs Ret':>8s} "
          f"{'Trades':>6s} {'Cost':>8s} {'Drag/yr':>8s}")
    print("  " + "-" * 124)

    for fid, label, _ in STRATEGIES:
        m = [r for r in all_results if r.factor_id == fid and r.rebal_freq == "monthly"]
        q = [r for r in all_results if r.factor_id == fid and r.rebal_freq == "quarterly"]
        if not m or not q:
            continue
        ms, qs = m[0].summary(), q[0].summary()
        print(f"  {label:<30s}  "
              f"{ms['Net Sharpe']:>7.2f} {ms['Net Ann Ret']:>+7.1%} "
              f"{ms['Gross Ann Ret']:>+7.1%} "
              f"{ms['Total Trades']:>5d} €{ms['Total Cost EUR']:>6,.0f} "
              f"{ms['Ann Cost Drag']:>+7.2%}"
              f"  |  "
              f"{qs['Net Sharpe']:>7.2f} {qs['Net Ann Ret']:>+7.1%} "
              f"{qs['Gross Ann Ret']:>+7.1%} "
              f"{qs['Total Trades']:>5d} €{qs['Total Cost EUR']:>6,.0f} "
              f"{qs['Ann Cost Drag']:>+7.2%}")

    # ---- Holdings frequency (top strategies) -----------------------------
    print("\n\n--- B1 HOLDINGS FREQUENCY (top 4 by Net Sharpe, quarterly) ---\n")
    q_results = sorted(
        [r for r in all_results if r.rebal_freq == "quarterly"],
        key=lambda r: r.summary()["Net Sharpe"], reverse=True)[:4]

    for r in q_results:
        freq = r.holding_frequency()
        total = len(r.periods)
        print(f"  {r.factor_label}:")
        for country, count in freq.most_common():
            bar = "█" * int(count / total * 30)
            print(f"    {country:20s} {count:3d}/{total} ({count/total:5.1%}) {bar}")
        print()

    # Save
    comp.to_csv(str(OUTPUT_DIR / "strategy_comparison.csv"), index=False)
    plot_comparison(all_results, OUTPUT_DIR / "all_strategies_comparison.png")
    for r in all_results:
        r.to_dataframe().to_csv(
            str(OUTPUT_DIR / f"returns_{r.factor_id}_{r.rebal_freq}.csv"))

    print(f"\nResults saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
