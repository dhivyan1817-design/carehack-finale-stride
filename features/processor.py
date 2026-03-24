import pandas as pd
import numpy as np


def compute_features(df, data_source="simulated"):
    """
    Computes progression-based structural features per monitoring zone.
    Source-agnostic: works for both simulated and live IoT data.

    Input columns expected:
        zone_id, timestamp, deformation_mm, gap_mm,
        vibration_level, temperature (optional), load_estimate
    """
    df = df.copy()
    df = df.sort_values(['zone_id', 'timestamp']).reset_index(drop=True)

    # Time delta in hours
    df['dt'] = (
        df.groupby('zone_id')['timestamp']
        .diff()
        .dt.total_seconds()
        .div(3600.0)
    )

    # Deformation growth rate (mm/hr)
    df['deformation_rate'] = (
        df.groupby('zone_id')['deformation_mm'].diff()
        / (df['dt'] + 1e-6)
    )

    # Acceleration of deformation (rate of change of rate)
    df['damage_acceleration'] = (
        df.groupby('zone_id')['deformation_rate'].diff()
        / (df['dt'] + 1e-6)
    )

    # Gap growth rate (mm/hr) — from ultrasonic
    df['gap_growth_rate'] = (
        df.groupby('zone_id')['gap_mm'].diff()
        / (df['dt'] + 1e-6)
    )

    # Rolling average vibration (smoothed over 3 readings)
    df['avg_vibration'] = (
        df.groupby('zone_id')['vibration_level']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Vibration sensitivity: how much deformation occurs per unit vibration
    df['vibration_sensitivity'] = (
        df['deformation_rate'] / (df['avg_vibration'] + 1e-6)
    )

    # Thermal effect — only available in simulated mode (no temp sensor on hardware yet)
    if data_source == "simulated" and 'temperature' in df.columns:
        df['temp_change'] = df.groupby('zone_id')['temperature'].diff()
        df['thermal_effect'] = df.groupby('zone_id', group_keys=False).apply(
            lambda x: x['deformation_mm'].diff() / (x['temperature'].diff() + 1e-6)
        ).reset_index(drop=True)
    else:
        df['thermal_effect'] = 0.0
        df['temp_change'] = 0.0

    # Ensure load_estimate exists
    if 'load_estimate' not in df.columns:
        df['load_estimate'] = 500.0

    df = df.fillna(0)
    return df
