"""
Emerging Markets Country-Level Factor Quintile Sort Backtester
==============================================================

Methodology:
    Each period, rank countries by a factor signal, assign to N equal-sized
    buckets (quintiles by default), then track realised forward returns per
    bucket. Bucket 1 = most attractive according to the signal.

Expected CSV formats:
    Returns  – columns: date, country, return
    Factor   – columns: date, country, factor_value

    Dates should be sortable strings (YYYY-MM-DD) or parseable by pandas.
    Returns are simple period returns (e.g. 0.03 for +3%).
    factor_value is the raw signal (the framework handles ranking direction).

Temporal alignment:
    The framework matches factor[t] → return[t+1] automatically.
    "t" is determined by the sorted unique dates in each dataset.

Usage:
    from factor_backtest import FactorBacktest, BacktestConfig

    cfg = BacktestConfig(n_buckets=5, higher_is_better=False,
                         factor_name="PE Ratio")
    bt = FactorBacktest(cfg)
    result = bt.run(factor_csv="factor.csv", returns_csv="returns.csv")
    result.print_summary()
    fig = result.plot_all()
    fig.savefig("pe_ratio_backtest.png", dpi=150, bbox_inches="tight")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Parameters
    ----------
    n_buckets : int
        Number of quantile buckets (default 5 = quintiles).
    higher_is_better : bool
        If True, high factor values → Bucket 1 (top).
        If False (default), low factor values → Bucket 1
        (e.g. low PE = cheap = attractive).
    min_countries : int
        Minimum number of countries with valid data in a period to form
        buckets. Periods with fewer are skipped.
    factor_name : str
        Label used in charts and tables.
    annualisation_factor : int
        Periods per year for annualising returns/vol (12 for monthly).
    """
    n_buckets: int = 5
    higher_is_better: bool = False
    min_countries: int = 5
    factor_name: str = "Factor"
    annualisation_factor: int = 12


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class BacktestResult:
    """Stores and analyses the output of a quintile-sort backtest."""

    def __init__(
        self,
        bucket_returns: pd.DataFrame,
        bucket_assignments: pd.DataFrame,
        config: BacktestConfig,
        skipped_dates: list,
    ):
        """
        Parameters
        ----------
        bucket_returns : DataFrame
            Index = date, columns = bucket labels (B1 .. Bn).
            Values are equal-weighted average returns of that bucket.
        bucket_assignments : DataFrame
            Columns: date, country, bucket, factor_value, fwd_return.
        config : BacktestConfig
        skipped_dates : list
            Dates that were skipped due to insufficient data.
        """
        self.bucket_returns = bucket_returns
        self.bucket_assignments = bucket_assignments
        self.config = config
        self.skipped_dates = skipped_dates

        # Derived series
        self.cumulative = (1 + self.bucket_returns).cumprod()
        self.long_short = (
            self.bucket_returns.iloc[:, 0] - self.bucket_returns.iloc[:, -1]
        )
        self.long_short.name = f"L/S (B1 − B{config.n_buckets})"
        self.cum_long_short = (1 + self.long_short).cumprod()

    # ---- Summary statistics ------------------------------------------------

    def summary_table(self) -> pd.DataFrame:
        """Annualised return, vol, Sharpe, hit-rate, max drawdown per bucket
        plus the long-short spread."""
        af = self.config.annualisation_factor

        def _row(series, label):
            ann_ret = (1 + series.mean()) ** af - 1
            ann_vol = series.std() * np.sqrt(af)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            hit = (series > 0).mean()
            cum = (1 + series).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            return {
                "Bucket": label,
                "Ann. Return": ann_ret,
                "Ann. Vol": ann_vol,
                "Sharpe": sharpe,
                "Hit Rate": hit,
                "Max DD": max_dd,
                "Periods": len(series),
            }

        rows = [_row(self.bucket_returns[c], c) for c in self.bucket_returns.columns]
        rows.append(_row(self.long_short, self.long_short.name))
        return pd.DataFrame(rows).set_index("Bucket")

    def monotonicity_score(self) -> float:
        """
        Fraction of adjacent bucket pairs where the higher-ranked bucket
        has a higher mean return.  1.0 = perfect monotonic decay from B1
        down to Bn.
        """
        means = self.bucket_returns.mean()
        n = len(means)
        if n < 2:
            return np.nan
        pairs_correct = sum(
            means.iloc[i] > means.iloc[i + 1] for i in range(n - 1)
        )
        return pairs_correct / (n - 1)

    def rank_ic(self) -> pd.Series:
        """
        Per-period Spearman rank correlation between factor value and
        forward return (Information Coefficient).

        Sign convention: if higher_is_better=False, we flip the factor sign
        so that a positive IC always means "the signal works as intended".
        """
        flip = -1.0 if not self.config.higher_is_better else 1.0
        ics = []
        for dt, grp in self.bucket_assignments.groupby("date"):
            if len(grp) < 4:
                continue
            ic = (grp["factor_value"] * flip).corr(
                grp["fwd_return"], method="spearman"
            )
            ics.append({"date": dt, "IC": ic})
        if not ics:
            return pd.Series(dtype=float, name="IC")
        return pd.DataFrame(ics).set_index("date")["IC"]

    def turnover(self) -> pd.Series:
        """
        Per-period fraction of countries that changed bucket vs. the
        previous period.
        """
        ba = self.bucket_assignments.sort_values(["country", "date"]).copy()
        ba["prev_bucket"] = ba.groupby("country")["bucket"].shift(1)
        ba = ba.dropna(subset=["prev_bucket"])
        ba["changed"] = (ba["bucket"] != ba["prev_bucket"]).astype(int)
        result = ba.groupby("date")["changed"].mean()
        result.name = "turnover"
        return result

    def rolling_long_short(self, window: int = 12) -> pd.Series:
        """Rolling annualised return of the long-short spread."""
        af = self.config.annualisation_factor
        rolling_mean = self.long_short.rolling(window).mean()
        return ((1 + rolling_mean) ** af - 1)

    # ---- Printing ----------------------------------------------------------

    def print_summary(self):
        """Pretty-print the summary table and key diagnostics."""
        tbl = self.summary_table()
        print("=" * 72)
        print(f"  FACTOR BACKTEST: {self.config.factor_name}")
        print(f"  Buckets: {self.config.n_buckets}  |  "
              f"Higher is better: {self.config.higher_is_better}  |  "
              f"Periods: {len(self.bucket_returns)}")
        if self.skipped_dates:
            print(f"  Skipped periods (insufficient data): "
                  f"{len(self.skipped_dates)}")
        print("=" * 72)
        print()

        # Format the table nicely
        fmt = tbl.copy()
        for c in ["Ann. Return", "Ann. Vol", "Max DD"]:
            fmt[c] = fmt[c].map(lambda x: f"{x:+.2%}")
        fmt["Hit Rate"] = fmt["Hit Rate"].map(lambda x: f"{x:.1%}")
        fmt["Sharpe"] = fmt["Sharpe"].map(lambda x: f"{x:.2f}")
        fmt["Periods"] = fmt["Periods"].astype(int)
        print(fmt.to_string())
        print()

        mono = self.monotonicity_score()
        print(f"  Monotonicity score:  {mono:.0%}  "
              f"({'perfect' if mono == 1.0 else 'imperfect'})")

        ic = self.rank_ic()
        if len(ic) > 0:
            ic_mean = ic.mean()
            ic_tstat = ic_mean / ic.std() * np.sqrt(len(ic))
            print(f"  Mean Rank IC:        {ic_mean:.3f}  "
                  f"(t = {ic_tstat:.2f})")
            print(f"  IC Hit Rate:         {(ic > 0).mean():.1%}")
        print(f"  Mean Turnover:       {self.turnover().mean():.1%}")
        print()

    # ---- Plotting ----------------------------------------------------------

    def plot_all(self, figsize: Tuple[int, int] = (16, 14)):
        """Generate a 2×2 panel of diagnostic charts."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Factor Backtest: {self.config.factor_name}",
            fontsize=15, fontweight="bold", y=0.98,
        )

        self._plot_cumulative(axes[0, 0])
        self._plot_bar(axes[0, 1])
        self._plot_long_short(axes[1, 0])
        self._plot_ic(axes[1, 1])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _bucket_color(self, i: int, n: int) -> str:
        cmap = plt.cm.RdYlGn_r
        return cmap(i / max(n - 1, 1))

    def _plot_cumulative(self, ax):
        n = self.config.n_buckets
        for i, col in enumerate(self.cumulative.columns):
            ax.plot(
                self.cumulative.index, self.cumulative[col],
                label=col, linewidth=1.8,
                color=self._bucket_color(i, n),
            )
        ax.set_title("Cumulative Returns by Bucket")
        ax.set_ylabel("Growth of $1")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_bar(self, ax):
        tbl = self.summary_table()
        tbl_b = tbl.iloc[: self.config.n_buckets]
        n = len(tbl_b)
        colors = [self._bucket_color(i, n) for i in range(n)]
        bars = ax.bar(
            tbl_b.index.astype(str), tbl_b["Ann. Return"],
            color=colors, edgecolor="black", linewidth=0.5,
        )
        ax.set_title("Annualised Return by Bucket")
        ax.set_ylabel("Ann. Return")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, tbl_b["Ann. Return"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:+.1%}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=9,
            )

    def _plot_long_short(self, ax):
        ax.plot(
            self.cum_long_short.index, self.cum_long_short.values,
            color="steelblue", linewidth=1.8,
        )
        ax.set_title(f"Cumulative L/S ({self.long_short.name})")
        ax.set_ylabel("Growth of $1")
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(True, alpha=0.3)

    def _plot_ic(self, ax):
        ic = self.rank_ic()
        if len(ic) == 0:
            ax.text(0.5, 0.5, "No IC data", transform=ax.transAxes,
                    ha="center")
            return
        ax.bar(ic.index, ic.values, color="grey", alpha=0.5, width=20)
        ax.axhline(
            ic.mean(), color="steelblue", linewidth=1.5,
            label=f"Mean IC = {ic.mean():.3f}",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Rank IC (Spearman)")
        ax.set_ylabel("IC")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class FactorBacktest:
    """
    Main entry point for running a quintile-sort factor backtest.

    Typical usage::

        from factor_backtest import FactorBacktest, BacktestConfig

        cfg = BacktestConfig(n_buckets=5, higher_is_better=False,
                             factor_name="PE Ratio")
        bt = FactorBacktest(cfg)
        result = bt.run(factor_csv="factor.csv", returns_csv="returns.csv")
        result.print_summary()
        result.plot_all()
        plt.show()
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    # ---- Data loading ------------------------------------------------------

    @staticmethod
    def _load_csv(
        path: Union[str, Path, pd.DataFrame],
        required_cols: list,
        date_col: str = "date",
    ) -> pd.DataFrame:
        if isinstance(path, pd.DataFrame):
            df = path.copy()
        else:
            df = pd.read_csv(path)

        df.columns = df.columns.str.strip().str.lower()
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        df[date_col] = pd.to_datetime(df[date_col])
        return df

    # ---- Core backtest -----------------------------------------------------

    def run(
        self,
        factor_csv: Union[str, Path, pd.DataFrame],
        returns_csv: Union[str, Path, pd.DataFrame],
        date_col: str = "date",
        country_col: str = "country",
        factor_col: str = "factor_value",
        return_col: str = "return",
    ) -> BacktestResult:
        """
        Execute the quintile-sort backtest.

        Parameters
        ----------
        factor_csv : path or DataFrame
            Factor signal data.  Required columns: [date, country, factor_value]
        returns_csv : path or DataFrame
            Realised return data.  Required columns: [date, country, return]

        Returns
        -------
        BacktestResult
        """
        cfg = self.config

        # Load & normalise
        factor_df = self._load_csv(
            factor_csv, [date_col, country_col, factor_col], date_col
        )
        returns_df = self._load_csv(
            returns_csv, [date_col, country_col, return_col], date_col
        )
        factor_df = factor_df.rename(columns={
            factor_col: "factor_value", country_col: "country", date_col: "date",
        })
        returns_df = returns_df.rename(columns={
            return_col: "return", country_col: "country", date_col: "date",
        })

        # Date sequences
        factor_dates = sorted(factor_df["date"].unique())
        return_dates_arr = np.array(sorted(returns_df["date"].unique()))

        bucket_return_rows = []
        assignment_rows = []
        skipped = []

        for fdate in factor_dates:
            # Forward return date: next available return date strictly after
            # the factor observation date
            candidates = return_dates_arr[return_dates_arr > fdate]
            if len(candidates) == 0:
                continue
            fwd_date = candidates[0]

            # Slices
            fslice = factor_df.loc[
                factor_df["date"] == fdate, ["country", "factor_value"]
            ].dropna(subset=["factor_value"])
            rslice = returns_df.loc[
                returns_df["date"] == fwd_date, ["country", "return"]
            ].dropna(subset=["return"])

            merged = fslice.merge(rslice, on="country", how="inner")
            if len(merged) < cfg.min_countries:
                skipped.append(fdate)
                continue

            # Rank: ascending=True puts lowest values first in the sort.
            # If higher_is_better=False, we want low values in B1,
            # so we sort ascending (low first → top of list → B1).
            # If higher_is_better=True, we sort descending.
            merged = merged.sort_values(
                "factor_value", ascending=not cfg.higher_is_better
            ).reset_index(drop=True)

            n = len(merged)
            bucket_size = n / cfg.n_buckets
            merged["bucket"] = [
                min(int(i // bucket_size) + 1, cfg.n_buckets) for i in range(n)
            ]

            # Bucket-level equal-weighted returns
            bkt_rets = merged.groupby("bucket")["return"].mean()
            row = {"date": fwd_date}
            for b in range(1, cfg.n_buckets + 1):
                row[f"B{b}"] = bkt_rets.get(b, np.nan)
            bucket_return_rows.append(row)

            for _, r in merged.iterrows():
                assignment_rows.append({
                    "date": fwd_date,
                    "country": r["country"],
                    "bucket": int(r["bucket"]),
                    "factor_value": r["factor_value"],
                    "fwd_return": r["return"],
                })

        if not bucket_return_rows:
            raise RuntimeError(
                "No valid periods produced.  Check date alignment, column "
                "names, and data coverage."
            )

        bucket_returns = (
            pd.DataFrame(bucket_return_rows).set_index("date").sort_index()
        )
        bucket_assignments = pd.DataFrame(assignment_rows)

        return BacktestResult(
            bucket_returns=bucket_returns,
            bucket_assignments=bucket_assignments,
            config=cfg,
            skipped_dates=skipped,
        )

    # ---- Convenience: combined CSV -----------------------------------------

    def run_combined(
        self,
        csv_path: Union[str, Path, pd.DataFrame],
        date_col: str = "date",
        country_col: str = "country",
        factor_col: str = "factor_value",
        return_col: str = "return",
    ) -> BacktestResult:
        """
        Run from a single CSV that has both factor and return columns.
        Alignment is still factor[t] → return[t+1].
        """
        df = self._load_csv(
            csv_path, [date_col, country_col, factor_col, return_col], date_col
        )
        return self.run(
            df[[date_col, country_col, factor_col]],
            df[[date_col, country_col, return_col]],
            date_col=date_col, country_col=country_col,
            factor_col=factor_col, return_col=return_col,
        )


# ---------------------------------------------------------------------------
# Utility: generate sample CSVs for testing
# ---------------------------------------------------------------------------

def generate_sample_data(
    n_countries: int = 15,
    n_months: int = 120,
    seed: int = 42,
    output_dir: str = ".",
) -> Tuple[str, str]:
    """
    Create synthetic factor and return CSVs for testing.

    The synthetic factor has weak predictive power by construction:
    countries with lower factor_value get a small return boost.

    Returns
    -------
    (factor_csv_path, returns_csv_path)
    """
    rng = np.random.default_rng(seed)

    countries = [
        "Brazil", "China", "India", "South Korea", "Taiwan",
        "South Africa", "Mexico", "Indonesia", "Turkey", "Thailand",
        "Poland", "Chile", "Malaysia", "Philippines", "Colombia",
    ][:n_countries]

    dates = pd.date_range("2014-01-31", periods=n_months, freq="ME")

    factor_rows = []
    return_rows = []

    base_pe = {c: rng.uniform(8, 25) for c in countries}

    for dt in dates:
        for c in countries:
            pe = base_pe[c] + rng.normal(0, 2)
            base_pe[c] = 0.95 * base_pe[c] + 0.05 * pe

            factor_rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "country": c,
                "factor_value": round(pe, 2),
            })

            # Small negative relationship with PE + large noise
            signal = -0.001 * (pe - 15)
            noise = rng.normal(0.005, 0.06)
            ret = signal + noise

            return_rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "country": c,
                "return": round(ret, 6),
            })

    factor_path = str(Path(output_dir) / "sample_factor.csv")
    returns_path = str(Path(output_dir) / "sample_returns.csv")

    pd.DataFrame(factor_rows).to_csv(factor_path, index=False)
    pd.DataFrame(return_rows).to_csv(returns_path, index=False)

    print(f"Generated: {factor_path}  ({len(factor_rows)} rows)")
    print(f"Generated: {returns_path} ({len(return_rows)} rows)")
    return factor_path, returns_path
