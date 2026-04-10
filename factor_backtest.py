"""
Emerging Markets Country-Level Factor Backtester (daily, long-only focus)
==========================================================================

Methodology:
    1. Factor signals are observed monthly (one per country per month-end).
    2. At each factor date t, rank countries and assign to N buckets.
    3. Hold each bucket from the trading day after t until the next factor date.
    4. Bucket equity curves are computed from DAILY country price returns
       (equal-weighted, daily rebalanced within each holding period).
    5. Stats (Sharpe, vol, drawdowns) are computed from daily returns,
       annualised using 252 trading days.

Temporal alignment (no look-ahead):
    Factor at date t  →  determines holdings starting the next trading day
    Holdings held until the next trading day after the next factor date

Long-only focus:
    Bucket 1 = most attractive countries (controlled by `higher_is_better`)
    Reports B1 vs MSCI EM benchmark (not L/S spread)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass
from typing import Optional, Union, List, Set, Dict
from pathlib import Path


# =========================================================================
# Country mapping (Bloomberg ticker → readable name)
# =========================================================================

COUNTRY_TICKERS = {
    "M1CNA Index":  "China",
    "MXIN Index":   "India",
    "MXKR Index":   "South Korea",
    "TAMSCI Index": "Taiwan",
    "MXID Index":   "Indonesia",
    "MXMY Index":   "Malaysia",
    "MXTH Index":   "Thailand",
    "MXPH Index":   "Philippines",
    "MXBR Index":   "Brazil",
    "MXMX Index":   "Mexico",
    "MXZA Index":   "South Africa",
    "MXTR Index":   "Turkey",
    "MXGR Index":   "Greece",
    "MXPL Index":   "Poland",
    "MXSA Index":   "Saudi Arabia",
}

BENCHMARK_TICKER = "MIMUEMRN Index"
BENCHMARK_NAME   = "MSCI EM"


# =========================================================================
# Chart style
# =========================================================================

PALETTE = {
    "green":     "#2ecc71",
    "lime":      "#82e0aa",
    "yellow":    "#f4d03f",
    "orange":    "#e67e22",
    "red":       "#e74c3c",
    "blue":      "#2980b9",
    "darkblue":  "#1a5276",
    "grey":      "#7f8c8d",
    "lightgrey": "#bdc3c7",
    "bg":        "#fafafa",
    "benchmark": "#2c3e50",
}

BUCKET_COLORS = [
    PALETTE["green"], PALETTE["lime"], PALETTE["yellow"],
    PALETTE["orange"], PALETTE["red"],
]


def apply_chart_style():
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     PALETTE["bg"],
        "axes.edgecolor":     PALETTE["lightgrey"],
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.color":         PALETTE["lightgrey"],
        "font.family":        "sans-serif",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "legend.fontsize":    9,
        "legend.framealpha":  0.9,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
    })


apply_chart_style()


# =========================================================================
# Daily price data loading
# =========================================================================

def load_daily_prices(csv_path):
    """
    Load daily prices and benchmark from EM_Indices_EUR.csv.

    Returns
    -------
    country_prices : DataFrame
        Daily prices indexed by date, columns = country names
    benchmark_prices : Series or None
        Daily MSCI EM prices (or None if not in file)
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    rename_map = {}
    for ticker, country in COUNTRY_TICKERS.items():
        if ticker in df.columns:
            rename_map[ticker] = country
        else:
            ticker_dot = ticker.replace(' ', '.')
            if ticker_dot in df.columns:
                rename_map[ticker_dot] = country

    if not rename_map:
        raise ValueError(
            f"No country tickers found in {csv_path}. "
            f"First 5 columns: {list(df.columns[:5])}"
        )

    country_prices = df[list(rename_map.keys())].rename(columns=rename_map)

    benchmark_prices = None
    for variant in [BENCHMARK_TICKER, BENCHMARK_TICKER.replace(' ', '.')]:
        if variant in df.columns:
            benchmark_prices = df[variant].rename(BENCHMARK_NAME)
            break

    return country_prices, benchmark_prices


def portfolio_daily_returns(daily_returns_df, holdings_history, daily_costs=None):
    """
    Compute daily portfolio returns from a holdings history.

    holdings_history : list of (start_date, end_date_exclusive, set_of_countries)
    daily_costs : dict {date: cost_pct} - subtracted from that day's return

    Equal-weighted across held countries each day (daily rebalanced).
    """
    period_returns_list = []

    for start, end, holdings in holdings_history:
        if end is None:
            mask = daily_returns_df.index >= start
        else:
            mask = (daily_returns_df.index >= start) & (daily_returns_df.index < end)

        period_data = daily_returns_df.loc[mask]
        if period_data.empty or not holdings:
            continue

        held_cols = [c for c in holdings if c in period_data.columns]
        if not held_cols:
            continue

        # Equal-weighted across countries (skipna handles missing prices)
        port_ret = period_data[held_cols].mean(axis=1, skipna=True)
        period_returns_list.append(port_ret)

    if not period_returns_list:
        return pd.Series(dtype=float)

    combined = pd.concat(period_returns_list).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    if daily_costs:
        for cost_date, cost_pct in daily_costs.items():
            if cost_date in combined.index:
                combined.loc[cost_date] -= cost_pct
            else:
                next_dates = combined.index[combined.index >= cost_date]
                if len(next_dates) > 0:
                    combined.loc[next_dates[0]] -= cost_pct

    return combined


def annualised_stats(daily_returns, periods_per_year=252):
    """Compute annualised stats from daily returns."""
    s = daily_returns.dropna()
    if len(s) == 0:
        return {}

    ann_ret = (1 + s.mean()) ** periods_per_year - 1
    ann_vol = s.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + s).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    return {
        "Ann Return": ann_ret,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Max DD": max_dd,
        "Total Return": cum.iloc[-1] - 1,
        "Hit Rate": (s > 0).mean(),
        "Days": len(s),
    }


def align_to_common_start(curves):
    """
    Align multiple equity curves to a common start date.
    Each curve is rebased to 1.0 at the latest first valid date.
    """
    if not curves:
        return curves

    first_dates = []
    for c in curves.values():
        valid = c.dropna()
        if len(valid) > 0:
            first_dates.append(valid.index[0])

    if not first_dates:
        return curves

    common_start = max(first_dates)

    aligned = {}
    for name, curve in curves.items():
        truncated = curve[curve.index >= common_start].dropna()
        if len(truncated) == 0:
            continue
        first_val = truncated.iloc[0]
        if first_val > 0:
            aligned[name] = truncated / first_val

    return aligned


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class BacktestConfig:
    n_buckets: int = 5
    higher_is_better: bool = False
    min_countries: int = 10
    factor_name: str = "Factor"
    periods_per_year: int = 252


# =========================================================================
# Result container
# =========================================================================

class BacktestResult:
    def __init__(self, bucket_assignments, daily_bucket_returns,
                 benchmark_daily_returns, config, skipped_dates=None):
        self.bucket_assignments = bucket_assignments
        self.daily_bucket_returns = daily_bucket_returns
        self.benchmark_daily_returns = benchmark_daily_returns
        self.config = config
        self.skipped_dates = skipped_dates or []

        # Equity curves
        self.bucket_equity = (1 + daily_bucket_returns.fillna(0)).cumprod()

        if benchmark_daily_returns is not None:
            bm_aligned = benchmark_daily_returns.reindex(daily_bucket_returns.index)
            self.benchmark_equity = (1 + bm_aligned.fillna(0)).cumprod()
        else:
            self.benchmark_equity = None

    def summary_table(self):
        """Per-bucket annualised stats, plus benchmark row."""
        rows = []
        for col in self.daily_bucket_returns.columns:
            s = annualised_stats(
                self.daily_bucket_returns[col],
                self.config.periods_per_year,
            )
            s["Bucket"] = col
            rows.append(s)

        if self.benchmark_daily_returns is not None:
            bm_aligned = self.benchmark_daily_returns.reindex(
                self.daily_bucket_returns.index)
            bm_stats = annualised_stats(bm_aligned, self.config.periods_per_year)
            bm_stats["Bucket"] = BENCHMARK_NAME
            rows.append(bm_stats)

        return pd.DataFrame(rows).set_index("Bucket")

    def b1_excess_stats(self):
        """B1 performance vs benchmark."""
        if self.benchmark_daily_returns is None:
            return {}

        b1 = self.daily_bucket_returns["B1"]
        bm = self.benchmark_daily_returns.reindex(b1.index)
        excess = (b1 - bm).dropna()
        if len(excess) == 0:
            return {}

        af = self.config.periods_per_year
        ann_excess = (1 + excess.mean()) ** af - 1
        te = excess.std() * np.sqrt(af)
        ir = ann_excess / te if te > 0 else np.nan

        return {
            "Excess Return": ann_excess,
            "Tracking Error": te,
            "Information Ratio": ir,
            "Hit Rate vs BM": (excess > 0).mean(),
        }

    def monotonicity_score(self):
        means = self.daily_bucket_returns.mean()
        n = len(means)
        if n < 2:
            return np.nan
        return sum(means.iloc[i] > means.iloc[i + 1] for i in range(n - 1)) / (n - 1)

    def rank_ic(self):
        if "fwd_return" not in self.bucket_assignments.columns:
            return pd.Series(dtype=float, name="IC")

        flip = -1.0 if not self.config.higher_is_better else 1.0
        ics = []
        for dt, grp in self.bucket_assignments.groupby("date"):
            if len(grp) < 4:
                continue
            ic = (grp["factor_value"] * flip).corr(
                grp["fwd_return"], method="spearman")
            ics.append({"date": dt, "IC": ic})

        if not ics:
            return pd.Series(dtype=float, name="IC")
        return pd.DataFrame(ics).set_index("date")["IC"]

    def turnover(self):
        ba = self.bucket_assignments.sort_values(["country", "date"]).copy()
        ba["prev_bucket"] = ba.groupby("country")["bucket"].shift(1)
        ba = ba.dropna(subset=["prev_bucket"])
        ba["changed"] = (ba["bucket"] != ba["prev_bucket"]).astype(int)
        return ba.groupby("date")["changed"].mean().rename("turnover")

    def print_summary(self):
        tbl = self.summary_table()
        print("=" * 72)
        print(f"  FACTOR: {self.config.factor_name}")
        print(f"  Buckets: {self.config.n_buckets}  |  "
              f"Higher is better: {self.config.higher_is_better}  |  "
              f"Days: {len(self.daily_bucket_returns)}  |  "
              f"Min countries: {self.config.min_countries}")
        if self.skipped_dates:
            print(f"  Skipped factor dates: {len(self.skipped_dates)}")
        print("=" * 72 + "\n")

        fmt = tbl.copy()
        for c in ["Ann Return", "Ann Vol", "Max DD", "Total Return"]:
            if c in fmt.columns:
                fmt[c] = fmt[c].map(lambda x: f"{x:+.2%}")
        if "Hit Rate" in fmt.columns:
            fmt["Hit Rate"] = fmt["Hit Rate"].map(lambda x: f"{x:.1%}")
        if "Sharpe" in fmt.columns:
            fmt["Sharpe"] = fmt["Sharpe"].map(lambda x: f"{x:.2f}")
        if "Days" in fmt.columns:
            fmt["Days"] = fmt["Days"].astype(int)
        print(fmt.to_string())
        print()

        ex = self.b1_excess_stats()
        if ex:
            print(f"  B1 vs {BENCHMARK_NAME}:")
            print(f"    Excess Return:     {ex['Excess Return']:+.2%}")
            print(f"    Tracking Error:    {ex['Tracking Error']:.2%}")
            print(f"    Information Ratio: {ex['Information Ratio']:.2f}")
            print(f"    Hit Rate vs BM:    {ex['Hit Rate vs BM']:.1%}")

        mono = self.monotonicity_score()
        print(f"  Monotonicity:      {mono:.0%}")
        ic = self.rank_ic()
        if len(ic) > 0:
            t = ic.mean() / ic.std() * np.sqrt(len(ic))
            print(f"  Mean Rank IC:      {ic.mean():.3f}  (t = {t:.2f})")
        to = self.turnover()
        if len(to) > 0:
            print(f"  Turnover:          {to.mean():.1%}")
        print()

    # ---- Plots ----

    def plot_all(self, figsize=(16, 13)):
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(self.config.factor_name, fontsize=18, fontweight="bold",
                     color=PALETTE["darkblue"], y=0.995)

        n_days = len(self.daily_bucket_returns)
        years = n_days / 252
        subtitle = (f"{self.config.n_buckets} quintile buckets  ·  "
                    f"~{years:.1f} years of daily data  ·  "
                    f"Min {self.config.min_countries} countries  ·  "
                    f"Long-only top bucket vs {BENCHMARK_NAME}")
        fig.text(0.5, 0.965, subtitle, ha="center", fontsize=11,
                 color=PALETTE["grey"])

        self._plot_cumulative_with_benchmark(axes[0, 0])
        self._plot_bar(axes[0, 1])
        self._plot_b1_vs_benchmark(axes[1, 0])
        self._plot_drawdowns(axes[1, 1])

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        return fig

    def _bucket_color(self, i, n):
        if n <= len(BUCKET_COLORS):
            return BUCKET_COLORS[i]
        return plt.cm.RdYlGn_r(i / max(n - 1, 1))

    def _plot_cumulative_with_benchmark(self, ax):
        n = self.config.n_buckets
        for i, col in enumerate(self.bucket_equity.columns):
            ax.plot(self.bucket_equity.index, self.bucket_equity[col],
                    label=col, linewidth=2.0, color=self._bucket_color(i, n))
        if self.benchmark_equity is not None:
            ax.plot(self.benchmark_equity.index, self.benchmark_equity,
                    label=BENCHMARK_NAME, linewidth=2.0,
                    color=PALETTE["benchmark"], linestyle="--", alpha=0.85)
        ax.set_title("Cumulative Returns by Bucket")
        ax.set_ylabel("Growth of €1")
        ax.legend(loc="upper left", frameon=True)

    def _plot_bar(self, ax):
        tbl = self.summary_table()
        bucket_rows = tbl[tbl.index.str.startswith("B")]
        n = len(bucket_rows)
        colors = [self._bucket_color(i, n) for i in range(n)]
        bars = ax.bar(bucket_rows.index.astype(str), bucket_rows["Ann Return"],
                      color=colors, edgecolor="white", linewidth=1.5,
                      width=0.65, zorder=3)

        if BENCHMARK_NAME in tbl.index:
            bm_ret = tbl.loc[BENCHMARK_NAME, "Ann Return"]
            ax.axhline(bm_ret, color=PALETTE["benchmark"], linewidth=1.8,
                       linestyle="--", label=f"{BENCHMARK_NAME} ({bm_ret:+.1%})",
                       zorder=4)
            ax.legend(loc="upper right", fontsize=9, frameon=True)

        ax.axhline(0, color="black", linewidth=0.8, zorder=2)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title("Annualised Return by Bucket")
        ax.set_ylabel("Ann. Return")

        for bar, val in zip(bars, bucket_rows["Ann Return"]):
            offset = 0.003 if val >= 0 else -0.003
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{val:+.1%}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=10, fontweight="bold")

    def _plot_b1_vs_benchmark(self, ax):
        b1 = self.bucket_equity["B1"]
        ax.plot(b1.index, b1.values, label="B1 (Top Bucket)",
                color=PALETTE["green"], linewidth=2.5)

        if self.benchmark_equity is not None:
            ax.plot(self.benchmark_equity.index, self.benchmark_equity,
                    label=BENCHMARK_NAME, color=PALETTE["benchmark"],
                    linewidth=2.0, linestyle="--", alpha=0.85)

            common_idx = b1.index.intersection(self.benchmark_equity.index)
            if len(common_idx) > 0:
                b1c = b1.reindex(common_idx)
                bmc = self.benchmark_equity.reindex(common_idx)
                ax.fill_between(common_idx, b1c, bmc,
                                where=(b1c >= bmc), alpha=0.18,
                                color=PALETTE["green"], interpolate=True)
                ax.fill_between(common_idx, b1c, bmc,
                                where=(b1c < bmc), alpha=0.18,
                                color=PALETTE["red"], interpolate=True)

        ax.set_title(f"B1 (Top Bucket) vs {BENCHMARK_NAME}")
        ax.set_ylabel("Growth of €1")
        ax.legend(loc="upper left", frameon=True)

    def _plot_drawdowns(self, ax):
        b1 = self.bucket_equity["B1"]
        dd_b1 = b1 / b1.cummax() - 1
        ax.fill_between(dd_b1.index, dd_b1, 0, alpha=0.4,
                        color=PALETTE["green"], label="B1")
        if self.benchmark_equity is not None:
            dd_bm = self.benchmark_equity / self.benchmark_equity.cummax() - 1
            ax.plot(dd_bm.index, dd_bm, color=PALETTE["benchmark"],
                    linewidth=1.8, linestyle="--", alpha=0.85,
                    label=BENCHMARK_NAME)
        ax.set_title("Drawdowns")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(loc="lower left", frameon=True)


# =========================================================================
# Engine
# =========================================================================

class FactorBacktest:
    def __init__(self, config=None):
        self.config = config or BacktestConfig()

    def run(self, factor_csv, daily_country_prices, benchmark_daily_prices=None):
        """
        Run quintile-sort backtest using monthly factor signals and daily prices.

        Parameters
        ----------
        factor_csv : str, Path, or DataFrame
            Monthly factor data with columns: date, country, factor_value
        daily_country_prices : DataFrame
            Daily country prices indexed by date, columns = country names
        benchmark_daily_prices : Series or None
            Daily benchmark prices

        Returns
        -------
        BacktestResult
        """
        cfg = self.config

        # Load factor
        if isinstance(factor_csv, pd.DataFrame):
            factor_df = factor_csv.copy()
        else:
            factor_df = pd.read_csv(factor_csv, parse_dates=["date"])
        factor_df.columns = factor_df.columns.str.strip().str.lower()
        factor_df["date"] = pd.to_datetime(factor_df["date"])
        factor_df = factor_df.dropna(subset=["factor_value"])

        # ==========================================================
        # MASTER DATA FIX: Alignment & Continuity
        # ==========================================================
        # 1. Snap all fragmented dates to the end of the calendar month
        factor_df["date"] = factor_df["date"].dt.to_period("M").dt.to_timestamp("M")
        
        # 2. Enforce the Global Common Start Date (Apples-to-Apples)
        global_start = pd.to_datetime("2006-01-31")
        factor_df = factor_df[factor_df["date"] >= global_start]
        # ==========================================================

        # Daily returns
        daily_returns = daily_country_prices.pct_change(fill_method=None)
        bm_daily_returns = (benchmark_daily_prices.pct_change(fill_method=None)
                            if benchmark_daily_prices is not None else None)

        factor_dates = sorted(factor_df["date"].unique())

        bucket_holdings = {b: [] for b in range(1, cfg.n_buckets + 1)}
        assignment_rows = []
        skipped = []

        for i, fdate in enumerate(factor_dates):
            # Step 1: factor values at this date
            fslice = factor_df[factor_df["date"] == fdate][
                ["country", "factor_value"]
            ].dropna(subset=["factor_value"])

            # Only countries with daily price coverage
            valid_cols = set(daily_returns.columns)
            fslice = fslice[fslice["country"].isin(valid_cols)]

            if len(fslice) < cfg.min_countries:
                skipped.append(fdate)
                continue

            # Step 2: holding period = next trading day after fdate
            #         until next trading day after the next factor date
            next_days = daily_returns.index[daily_returns.index > fdate]
            if len(next_days) == 0:
                continue
            start_date = next_days[0]

            if i + 1 < len(factor_dates):
                next_fdate = factor_dates[i + 1]
                next_next = daily_returns.index[daily_returns.index > next_fdate]
                end_date = next_next[0] if len(next_next) > 0 else None
            else:
                end_date = None

            # Step 3: rank and assign buckets
            fslice = fslice.sort_values(
                "factor_value", ascending=not cfg.higher_is_better
            ).reset_index(drop=True)

            n = len(fslice)
            bucket_size = n / cfg.n_buckets
            fslice["bucket"] = [
                min(int(j // bucket_size) + 1, cfg.n_buckets) for j in range(n)
            ]

            # Add to each bucket's holdings history
            for bucket in range(1, cfg.n_buckets + 1):
                bucket_countries = set(fslice[fslice["bucket"] == bucket]["country"])
                bucket_holdings[bucket].append((start_date, end_date, bucket_countries))

            # Step 4: per-country forward returns over the holding period (for IC)
            if end_date is not None:
                period_mask = ((daily_returns.index >= start_date)
                               & (daily_returns.index < end_date))
            else:
                period_mask = daily_returns.index >= start_date

            period_data = daily_returns.loc[period_mask]
            if len(period_data) > 0:
                period_total = (1 + period_data.fillna(0)).prod() - 1
            else:
                period_total = pd.Series(dtype=float)

            for _, row in fslice.iterrows():
                assignment_rows.append({
                    "date": fdate,
                    "country": row["country"],
                    "bucket": int(row["bucket"]),
                    "factor_value": row["factor_value"],
                    "fwd_return": period_total.get(row["country"], np.nan),
                })

        if not assignment_rows:
            raise RuntimeError(
                "No valid periods produced. Check factor data coverage."
            )

        # Step 5: build daily bucket return series
        bucket_daily = {}
        for bucket, history in bucket_holdings.items():
            port = portfolio_daily_returns(daily_returns, history)
            bucket_daily[f"B{bucket}"] = port

        daily_bucket_returns = pd.DataFrame(bucket_daily).sort_index()
        bucket_assignments = pd.DataFrame(assignment_rows)

        return BacktestResult(
            bucket_assignments=bucket_assignments,
            daily_bucket_returns=daily_bucket_returns,
            benchmark_daily_returns=bm_daily_returns,
            config=cfg,
            skipped_dates=skipped,
        )


# =========================================================================
# Eligibility check
# =========================================================================

def check_factor_eligibility(factor_path, min_countries=10, min_eligible_periods=36):
    df = pd.read_csv(factor_path, parse_dates=["date"])
    df = df.dropna(subset=["factor_value"])

    counts = df.groupby("date")["country"].nunique()
    eligible_dates = (counts >= min_countries).sum()

    info = {
        "n_dates": len(counts),
        "eligible_dates": int(eligible_dates),
        "median_countries": float(counts.median()) if len(counts) > 0 else 0,
        "min_countries_in_data": int(counts.min()) if len(counts) > 0 else 0,
        "max_countries_in_data": int(counts.max()) if len(counts) > 0 else 0,
        "n_unique_countries": df["country"].nunique(),
    }

    if info["max_countries_in_data"] < min_countries:
        info["eligible"] = False
        info["reason"] = (f"Never reaches {min_countries} countries "
                          f"(max={info['max_countries_in_data']})")
    elif eligible_dates < min_eligible_periods:
        info["eligible"] = False
        info["reason"] = (f"Only {eligible_dates} dates with ≥{min_countries} "
                          f"countries (need {min_eligible_periods})")
    else:
        info["eligible"] = True
        info["reason"] = "OK"

    return info
