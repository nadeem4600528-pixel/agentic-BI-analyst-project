"""Generate a realistic multi-year sales dataset for demos and tests.

Creates a CSV/JSON of order records spanning ~3 years with regions,
categories, customer segments, quantity, sales, discount and profit — enough
variety to drive trends, KPIs, breakdowns, distributions and insights.

Usage:
    python scripts/generate_sample_data.py [rows] [output_path]
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Furniture", "Office Supplies", "Software"]
    segments = ["Consumer", "Corporate", "Home Office"]
    base_prices = {"Electronics": 520, "Furniture": 340, "Office Supplies": 48, "Software": 160}

    records = []
    start = datetime(2022, 1, 1)
    for i in range(rows):
        # Spread across ~3 years and add gentle upward growth.
        day_offset = int(rng.integers(0, 1095))
        order_date = start + timedelta(days=day_offset)
        growth = 1 + (day_offset / 1095) * 0.35  # ~35% growth over the window

        region = rng.choice(regions, p=[0.30, 0.25, 0.25, 0.20])
        category = rng.choice(categories, p=[0.40, 0.25, 0.20, 0.15])
        segment = rng.choice(segments, p=[0.50, 0.35, 0.15])

        quantity = int(rng.integers(1, 12))
        price = base_prices[category]
        sales = quantity * price * float(rng.normal(1.0, 0.18)) * growth

        discount = float(rng.choice([0, 0.05, 0.10, 0.15, 0.20], p=[0.50, 0.20, 0.12, 0.10, 0.08]))
        sales *= 1 - discount
        margin = float(rng.normal(0.24, 0.06)) - discount * 0.15
        profit = sales * margin

        records.append({
            "order_id": f"ORD-{i:05d}",
            "order_date": order_date.date().isoformat(),
            "region": str(region),
            "category": str(category),
            "segment": str(segment),
            "quantity": quantity,
            "sales": round(max(sales, 1.0), 2),
            "discount": round(discount, 2),
            "profit": round(profit, 2),
        })

    df = pd.DataFrame(records)
    # Sprinkle a few missing values and big-ticket outliers to look real.
    miss_idx = rng.choice(df.index, size=max(1, rows // 80), replace=False)
    df.loc[miss_idx, "discount"] = np.nan
    out_idx = rng.choice(df.index, size=max(1, rows // 200), replace=False)
    df.loc[out_idx, "sales"] = df.loc[out_idx, "sales"] * rng.uniform(4, 7, size=len(out_idx))
    return df


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "test_data" / "sample_sales.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(rows)
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame)} rows to {out}")
