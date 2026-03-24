"""
Generates synthetic time-series structural health data for a bridge.
Each zone models a distinct physical location with realistic damage behavior.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Bridge monitoring zones — realistic structural locations
BRIDGE_ZONES = {
    "North-Abutment": {
        "desc": "Foundation zone at northern end",
        "base_deformation": 8.0,
        "base_gap": 1.2,
        "growth_rate": 0.09,       # High — deteriorating abutment
        "acceleration": 0.007,
        "stabilizes": False,
    },
    "Main-Span-Center": {
        "desc": "Central span mid-point, highest load",
        "base_deformation": 12.0,
        "base_gap": 2.1,
        "growth_rate": 0.05,
        "acceleration": 0.003,
        "stabilizes": False,
    },
    "East-Pier-Joint": {
        "desc": "Beam-to-pier connection on east side",
        "base_deformation": 6.0,
        "base_gap": 0.9,
        "growth_rate": 0.04,
        "acceleration": 0.0,
        "stabilizes": True,        # Was repaired — risk declining
    },
    "West-Pier-Joint": {
        "desc": "Beam-to-pier connection on west side",
        "base_deformation": 5.0,
        "base_gap": 0.7,
        "growth_rate": 0.015,
        "acceleration": 0.0,
        "stabilizes": False,
    },
    "South-Abutment": {
        "desc": "Foundation zone at southern end",
        "base_deformation": 7.0,
        "base_gap": 1.0,
        "growth_rate": 0.02,
        "acceleration": 0.0,
        "stabilizes": False,
    },
}


def generate_data(days=30, interval_hours=4, seed=42):
    np.random.seed(seed)
    data = []
    base_time = datetime(2024, 1, 1)
    steps = days * 24 // interval_hours
    mid = steps // 2

    for zone_id, cfg in BRIDGE_ZONES.items():
        deformation = cfg["base_deformation"]
        gap = cfg["base_gap"]

        for step in range(steps):
            t = base_time + timedelta(hours=step * interval_hours)

            # Environment
            temp = 18 + 12 * np.sin(2 * np.pi * step * interval_hours / 24) + np.random.normal(0, 1.5)
            vibration = np.random.uniform(0.1, 0.6)
            load = np.random.uniform(200, 1200)

            # Growth physics
            if cfg["stabilizes"] and step > mid:
                effective_growth = -0.008 * np.random.uniform(0.5, 1.5)
            else:
                base_g = cfg["growth_rate"] + cfg["acceleration"] * step
                effective_growth = base_g * (1 + 0.08 * (load / 1200) + 0.15 * vibration)

            deformation += effective_growth + np.random.normal(0, 0.004)
            gap += max(0, effective_growth) * 0.12 + np.random.normal(0, 0.002)

            data.append({
                'zone_id': zone_id,
                'zone_description': cfg["desc"],
                'timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'deformation_mm': round(max(1.0, deformation), 4),
                'gap_mm': round(max(0.05, gap), 4),
                'vibration_level': round(vibration, 4),
                'temperature': round(temp, 2),
                'load_estimate': round(load, 1),
            })

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Generating synthetic bridge structural data...")
    df = generate_data()

    out = "data/simulated/structural_data.csv"
    os.makedirs("data/simulated", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")
    print(df.groupby('zone_id')[['deformation_mm', 'gap_mm']].max())
