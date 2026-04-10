import pandas as pd
from pathlib import Path

DATA_DIR = Path("backtest_data")
registry = pd.read_csv(DATA_DIR / "factor_registry.csv")
GLOBAL_START = pd.to_datetime("2006-01-31")

print("=" * 90)
print("  THE MONTH-END SNAPPING DIAGNOSTIC")
print("  Forcing all fragmented dates to the end of the calendar month.")
print("=" * 90)

results = []

for _, row in registry.iterrows():
    factor_file = DATA_DIR / row["filename"]
    if not factor_file.exists():
        continue

    df = pd.read_csv(factor_file, parse_dates=["date"])
    df = df.dropna(subset=["factor_value"])
    
    # ---> THE MAGIC FIX <---
    # This forces every date to the absolute last day of that month
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
    
    df = df[df["date"] >= GLOBAL_START]
    
    if df.empty:
        continue

    counts = df.groupby("date")["country"].nunique()
    total_months = len(counts)
    invalid_months = (counts < 10).sum()
    missing_pct = (invalid_months / total_months) * 100
    
    results.append({
        "Factor": row["label"],
        "Total": total_months,
        "Invalid": invalid_months,
        "Missing %": missing_pct
    })

results = sorted(results, key=lambda x: x["Missing %"])

print(f"{'Factor Name':<30} | {'Total':<5} | {'Invalid':<7} | {'Miss %':<6}")
print("-" * 90)
for r in results:
    status = "🟢" if r["Missing %"] < 5 else "🔴"
    print(f"{status} {r['Factor']:<28} | {r['Total']:<5} | {r['Invalid']:<7} | {r['Missing %']:>5.1f}%")
print("=" * 90)