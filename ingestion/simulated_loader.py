import pandas as pd
import os


REQUIRED_COLS = [
    'zone_id', 'deformation_mm', 'gap_mm',
    'vibration_level', 'load_estimate', 'timestamp'
]


def load_structural_data(filepath):
    """
    Loads zone-based structural data from CSV.
    Validates required columns and normalizes types.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}' in {filepath}")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['zone_id', 'timestamp']).reset_index(drop=True)
    return df
