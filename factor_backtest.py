"""
Emerging Markets Country-Level Factor Quintile Sort Backtester
==============================================================

Methodology:
    Each period t, rank all countries by a factor signal observed at t,
    assign to N equal-sized buckets. Bucket 1 = most attractive.
    Measure each bucket's equal-weighted return over month t+1.
    Repeat every month.

Temporal alignment:
    factor observed at date t → forward return = the return measured at
    the NEXT available return date after t.  This ensures no look-ahead
    bias: we rank on information available today and measure what happens
    next month.

Expected CSV formats:
    Factor:   date, country, factor_value
    Returns:  date, country, return   (simple monthly return, e.g. 0.03 = +3%)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from pathlib import Path


# =========================================================================
# Chart style — presentation quality
# =========================================================================

PALETTE = {
    "green":    "#2ecc71",
    "lime":     "#82e0aa",
    "yellow":   "#f4d03f",
    "orange":   "#e67e22",
    "red":      "#e74c3c",
    "blue":     "#2980b9",
    "darkblue": "#1a5276",
    "grey":     "#7f8c8d",
    "lightgrey":"#bdc3c7",
    "bg":       "#fafafa",
}

BUCKET_COLORS = [
    PALETTE["green"], PALETTE["lime"], PALETTE["yellow"],
    PALETTE["orange"], PALETTE["red"],
]

def apply_chart_style():
    """Set global matplotlib style for presentation-quality output."""
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
# Configuration
# =========================================================================

@dataclass
class BacktestConfig:
    n_buckets: int = 5
    higher_is_better: bool = False
    min_countries: int = 10      # need ≥2 per bucket for meaningful sort
    factor_name: str = "Factor"
    annualisation_factor: int = 12


# =========================================================================
# Result container
# =========================================================================

class BacktestResult:
    def __init__(self, bucket_returns, bucket_assignments, config, skipped_dates):
        self.bucket_returns = bucket_returns
        self.bucket_assignments = bucket_assignments
        self.config = config
        self.skipped_dates = skipped_dates

        self.cumulative = (1 + self.bucket_returns).cumprod()
        self.long_short = (
            self.bucket_returns.iloc[:, 0] - self.bucket_returns.iloc[:, -1]
        )
        self.long_short.name = f"L/S (B1 − B{config.n_buckets})"
        self.cum_long_short = (1 + self.long_short).cumprod()

    def summary_table(self) -> pd.DataFrame:
        af = self.config.annualisation_factor

        def _row(series, label):
            ann_ret = (1 + series.mean()) ** af - 1
            ann_vol = series.std() * np.sqrt(af)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            hit = (series > 0).mean()
            cum = (1 + series).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            return {
                "Bucket": label, "Ann. Return": ann_ret, "Ann. Vol": ann_vol,
                "Sharpe": sharpe, "Hit Rate": hit, "Max DD": max_dd,
                "Periods": len(series),
            }

        rows = [_row(self.bucket_returns[c], c) for c in self.bucket_returns.columns]
        rows.append(_row(self.long_short, self.long_short.name))
        return pd.DataFrame(rows).set_index("Bucket")

    def monotonicity_score(self) -> float:
        means = self.bucket_returns.mean()
        n = len(means)
        if n < 2:
            return np.nan
        return sum(means.iloc[i] > means.iloc[i + 1] for i in range(n - 1)) / (n - 1)

    def rank_ic(self) -> pd.Series:
        flip = -1.0 if not self.config.higher_is_better else 1.0
        ics = []
        for dt, grp in self.bucket_assignments.groupby("date"):
            if len(grp) < 4:
                continue
            ic = (grp["factor_value"] * flip).corr(grp["fwd_return"], method="spearman")
            ics.append({"date": dt, "IC": ic})
        if not ics:
            return pd.Series(dtype=float, name="IC")
        return pd.DataFrame(ics).set_index("date")["IC"]

    def turnover(self) -> pd.Series:
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
              f"Periods: {len(self.bucket_returns)}  |  "
              f"Min countries: {self.config.min_countries}")
        if self.skipped_dates:
            print(f"  Skipped (insufficient data): {len(self.skipped_dates)}")
        print("=" * 72)
        print()

        fmt = tbl.copy()
        for c in ["Ann. Return", "Ann. Vol", "Max DD"]:
            fmt[c] = fmt[c].map(lambda x: f"{x:+.2%}")
        fmt["Hit Rate"] = fmt["Hit Rate"].map(lambda x: f"{x:.1%}")
        fmt["Sharpe"] = fmt["Sharpe"].map(lambda x: f"{x:.2f}")
        fmt["Periods"] = fmt["Periods"].astype(int)
        print(fmt.to_string())
        print()

        mono = self.monotonicity_score()
        print(f"  Monotonicity: {mono:.0%}")
        ic = self.rank_ic()
        if len(ic) > 0:
            t = ic.mean() / ic.std() * np.sqrt(len(ic))
            print(f"  Mean Rank IC: {ic.mean():.3f}  (t = {t:.2f})")
            print(f"  IC Hit Rate:  {(ic > 0).mean():.1%}")
        print(f"  Turnover:     {self.turnover().mean():.1%}\n")

    # ---- Presentation-quality plots --------------------------------------

    def plot_all(self, figsize=(16, 13)):
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            self.config.factor_name,
            fontsize=18, fontweight="bold", y=0.995,
            color=PALETTE["darkblue"],
        )

        subtitle = (f"{self.config.n_buckets} quintile buckets  ·  "
                     f"{len(self.bucket_returns)} months  ·  "
                     f"Min {self.config.min_countries} countries")
        fig.text(0.5, 0.965, subtitle, ha="center", fontsize=11,
                 color=PALETTE["grey"])

        self._plot_cumulative(axes[0, 0])
        self._plot_bar(axes[0, 1])
        self._plot_long_short(axes[1, 0])
        self._plot_ic(axes[1, 1])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def _get_color(self, i, n):
        if n <= len(BUCKET_COLORS):
            return BUCKET_COLORS[i]
        cmap = plt.cm.RdYlGn_r
        return cmap(i / max(n - 1, 1))

    def _plot_cumulative(self, ax):
        n = self.config.n_buckets
        for i, col in enumerate(self.cumulative.columns):
            ax.plot(self.cumulative.index, self.cumulative[col],
                    label=col, linewidth=2.0, color=self._get_color(i, n))
        ax.set_title("Cumulative Returns by Bucket")
        ax.set_ylabel("Growth of €1")
        ax.legend(loc="upper left", frameon=True)

    def _plot_bar(self, ax):
        tbl = self.summary_table().iloc[:self.config.n_buckets]
        n = len(tbl)
        colors = [self._get_color(i, n) for i in range(n)]
        bars = ax.bar(tbl.index.astype(str), tbl["Ann. Return"],
                      color=colors, edgecolor="white", linewidth=1.5,
                      width=0.65, zorder=3)
        ax.set_title("Annualised Return by Bucket")
        ax.set_ylabel("Ann. Return")
        ax.axhline(0, color="black", linewidth=0.8, zorder=2)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        for bar, val in zip(bars, tbl["Ann. Return"]):
            offset = 0.003 if val >= 0 else -0.003
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{val:+.1%}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=10, fontweight="bold")

    def _plot_long_short(self, ax):
        ax.plot(self.cum_long_short.index, self.cum_long_short.values,
                color=PALETTE["blue"], linewidth=2.0)
        ax.fill_between(self.cum_long_short.index, 1,
                        self.cum_long_short.values,
                        where=self.cum_long_short.values >= 1,
                        alpha=0.15, color=PALETTE["green"])
        ax.fill_between(self.cum_long_short.index, 1,
                        self.cum_long_short.values,
                        where=self.cum_long_short.values < 1,
                        alpha=0.15, color=PALETTE["red"])
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Cumulative Long / Short  ({self.long_short.name})")
        ax.set_ylabel("Growth of €1")

    def _plot_ic(self, ax):
        ic = self.rank_ic()
        if len(ic) == 0:
            ax.text(0.5, 0.5, "No IC data", transform=ax.transAxes, ha="center")
            return
        colors = [PALETTE["green"] if v > 0 else PALETTE["red"] for v in ic.values]
        ax.bar(ic.index, ic.values, color=colors, alpha=0.5, width=25)
        ax.axhline(ic.mean(), color=PALETTE["darkblue"], linewidth=2,
                    label=f"Mean IC = {ic.mean():.3f}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Rank IC  (Spearman)")
        ax.set_ylabel("IC")
        ax.legend(frameon=True)


# =========================================================================
# Main engine
# =========================================================================

class FactorBacktest:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    @staticmethod
    def _load_csv(path, required_cols, date_col="date"):
        if isinstance(path, pd.DataFrame):
            df = path.copy()
        else:
            df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Have: {list(df.columns)}")
        df[date_col] = pd.to_datetime(df[date_col])
        return df

    def run(self, factor_csv, returns_csv,
            date_col="date", country_col="country",
            factor_col="factor_value", return_col="return") -> BacktestResult:
        """
        Execute the quintile-sort backtest.

        For each factor date t:
          1. Get factor values for all countries at t.
          2. Find the next return date t+1 (strictly after t).
          3. Get country returns at t+1.
          4. Inner-join: only countries with both factor[t] and return[t+1].
          5. If ≥ min_countries: rank, assign buckets, record bucket returns.
          6. Otherwise: skip this period.
        """
        cfg = self.config

        factor_df = self._load_csv(factor_csv, [date_col, country_col, factor_col], date_col)
        returns_df = self._load_csv(returns_csv, [date_col, country_col, return_col], date_col)

        factor_df = factor_df.rename(columns={
            factor_col: "factor_value", country_col: "country", date_col: "date"})
        returns_df = returns_df.rename(columns={
            return_col: "return", country_col: "country", date_col: "date"})

        factor_dates = sorted(factor_df["date"].unique())
        return_dates_arr = np.array(sorted(returns_df["date"].unique()))

        bucket_return_rows = []
        assignment_rows = []
        skipped = []

        for fdate in factor_dates:
            # Step 1-2: find next return date strictly after factor date
            candidates = return_dates_arr[return_dates_arr > fdate]
            if len(candidates) == 0:
                continue
            fwd_date = candidates[0]

            # Step 3-4: merge factor[t] with return[t+1] on country
            fslice = factor_df.loc[
                factor_df["date"] == fdate, ["country", "factor_value"]
            ].dropna(subset=["factor_value"])
            rslice = returns_df.loc[
                returns_df["date"] == fwd_date, ["country", "return"]
            ].dropna(subset=["return"])
            merged = fslice.merge(rslice, on="country", how="inner")

            # Step 5: check minimum coverage
            if len(merged) < cfg.min_countries:
                skipped.append(fdate)
                continue

            # Rank: sort so B1 = most attractive
            merged = merged.sort_values(
                "factor_value", ascending=not cfg.higher_is_better
            ).reset_index(drop=True)

            n = len(merged)
            bucket_size = n / cfg.n_buckets
            merged["bucket"] = [
                min(int(j // bucket_size) + 1, cfg.n_buckets) for j in range(n)
            ]

            # Record bucket-level equal-weighted returns
            bkt_rets = merged.groupby("bucket")["return"].mean()
            row = {"date": fwd_date}
            for b in range(1, cfg.n_buckets + 1):
                row[f"B{b}"] = bkt_rets.get(b, np.nan)
            bucket_return_rows.append(row)

            for _, r in merged.iterrows():
                assignment_rows.append({
                    "date": fwd_date, "country": r["country"],
                    "bucket": int(r["bucket"]),
                    "factor_value": r["factor_value"],
                    "fwd_return": r["return"],
                })

        if not bucket_return_rows:
            raise RuntimeError(
                "No valid periods. Check date alignment and data coverage.")

        bucket_returns = pd.DataFrame(bucket_return_rows).set_index("date").sort_index()
        bucket_assignments = pd.DataFrame(assignment_rows)

        return BacktestResult(bucket_returns, bucket_assignments, cfg, skipped)

    def run_combined(self, csv_path, date_col="date", country_col="country",
                     factor_col="factor_value", return_col="return"):
        df = self._load_csv(csv_path, [date_col, country_col, factor_col, return_col], date_col)
        return self.run(
            df[[date_col, country_col, factor_col]],
            df[[date_col, country_col, return_col]],
            date_col=date_col, country_col=country_col,
            factor_col=factor_col, return_col=return_col)


# =========================================================================
# Eligibility check (used by run_all_backtests.py)
# =========================================================================

def check_factor_eligibility(factor_path: Union[str, Path],
                             min_countries: int = 10,
                             min_eligible_periods: int = 36) -> dict:
    """
    Pre-screen a factor file for data quality.

    Returns dict with:
      eligible: bool
      reason: str (why not eligible, if applicable)
      n_dates, median_countries, min_countries_in_data, etc.
    """
    df = pd.read_csv(factor_path, parse_dates=["date"])
    df = df.dropna(subset=["factor_value"])

    counts = df.groupby("date")["country"].nunique()
    eligible_dates = (counts >= min_countries).sum()

    info = {
        "n_dates": len(counts),
        "eligible_dates": int(eligible_dates),
        "median_countries": float(counts.median()),
        "min_countries_in_data": int(counts.min()),
        "max_countries_in_data": int(counts.max()),
        "unique_countries": sorted(df["country"].unique().tolist()),
        "n_unique_countries": df["country"].nunique(),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
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
