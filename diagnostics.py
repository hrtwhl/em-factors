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


import pandas as pd
from pathlib import Path

DATA_DIR = Path("backtest_data")
registry = pd.read_csv(DATA_DIR / "factor_registry.csv")

print("=" * 60)
print("FACTOR START DATES DIAGNOSTIC")
print("=" * 60)

start_dates = {}

for _, row in registry.iterrows():
    factor_file = DATA_DIR / row["filename"]
    if factor_file.exists():
        df = pd.read_csv(factor_file, parse_dates=["date"])
        df = df.dropna(subset=["factor_value"])
        if not df.empty:
            earliest_date = df["date"].min()
            start_dates[row["label"]] = earliest_date

# Sort by start date
sorted_dates = sorted(start_dates.items(), key=lambda x: x[1])

print(f"{'Factor Name':<35} | {'Start Date'}")
print("-" * 60)
for name, date in sorted_dates:
    print(f"{name:<35} | {date.date()}")

print("-" * 60)
earliest_overall = sorted_dates[0][1]
latest_start = sorted_dates[-1][1]

print(f"\nOldest factor starts: {earliest_overall.date()}")
print(f"Youngest factor starts: {latest_start.date()}")

# The "Global Common Timeframe" starts on the youngest factor's start date
print(f"\nIf we enforce a strict GLOBAL COMMON TIMEFRAME:")
print(f"We must start all backtests on: {latest_start.date()}")
years_lost = (latest_start - earliest_overall).days / 365.25
print(f"-> This means throwing away up to {years_lost:.1f} years of history from the oldest factors.")


import pandas as pd
from pathlib import Path

DATA_DIR = Path("backtest_data")
registry = pd.read_csv(DATA_DIR / "factor_registry.csv")

print("=" * 70)
print("  STRICT CONTINUITY DIAGNOSTIC")
print("  Rule: Trim early sparse data. Zero tolerance for mid-stream gaps.")
print("=" * 70)

survivors = []
casualties = []

for _, row in registry.iterrows():
    factor_file = DATA_DIR / row["filename"]
    if not factor_file.exists():
        continue
        
    df = pd.read_csv(factor_file, parse_dates=["date"])
    df = df.dropna(subset=["factor_value"])
    
    if df.empty:
        continue

    # Count countries per date
    counts = df.groupby("date")["country"].nunique()
    
    # Find the first date where we have >= 10 countries
    valid_dates = counts[counts >= 10].index
    
    if len(valid_dates) == 0:
        casualties.append({
            "Factor": row["label"], 
            "Reason": "NEVER reached 10 countries."
        })
        continue
        
    true_start_date = valid_dates.min()
    
    # Look at all data strictly AFTER the true start date
    continuous_period = counts[counts.index >= true_start_date]
    
    # Did it ever dip below 10 again?
    min_countries_after_start = continuous_period.min()
    
    if min_countries_after_start < 10:
        # Find the exact date it broke to show the user
        broken_date = continuous_period[continuous_period < 10].index[0]
        casualties.append({
            "Factor": row["label"], 
            "Reason": f"Dropped to {min_countries_after_start} countries on {broken_date.date()}"
        })
    else:
        survivors.append({
            "Factor": row["label"],
            "Start Date": true_start_date,
            "Total Months": len(continuous_period)
        })

# --- Print Results ---
survivors = sorted(survivors, key=lambda x: x["Start Date"])

print(f"\n✅ SURVIVORS ({len(survivors)} factors)")
print("-" * 70)
print(f"{'Factor Name':<35} | {'Continuous Start Date':<22} | {'Months'}")
print("-" * 70)
for s in survivors:
    print(f"{s['Factor']:<35} | {str(s['Start Date'].date()):<22} | {s['Total Months']}")

print(f"\n❌ CASUALTIES ({len(casualties)} factors dropped)")
print("-" * 70)
for c in casualties:
    print(f"{c['Factor']:<35} | {c['Reason']}")
print("=" * 70)