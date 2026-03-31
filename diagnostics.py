import pandas as pd
from pathlib import Path

data_dir = Path("backtest_data")

for f in sorted(data_dir.glob("factor_*.csv")):
    if f.name == "factor_registry.csv":
        continue
    df = pd.read_csv(f, parse_dates=["date"])
    
    # Per-date country count
    counts = df.groupby("date")["country"].nunique()
    
    print(f"\n{f.name}:")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Countries per date: min={counts.min()}, median={counts.median():.0f}, max={counts.max()}")
    print(f"  Unique countries: {sorted(df['country'].unique())}")
    
    if counts.min() < 10:
        thin_dates = counts[counts < 10]
        print(f"  WARNING: {len(thin_dates)} dates have fewer than 10 countries")