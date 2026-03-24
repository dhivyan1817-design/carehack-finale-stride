"""
STRIDE — Structural Time-based Risk Intelligence for Damage Evolution
Main pipeline: generates data → features → trains model → saves results
Run this once before launching the dashboard.
"""

import os
import subprocess
import sys

from ingestion.simulated_loader import load_structural_data
from features.processor import compute_features
from model.risk_model import RiskModel
from decision.logic import apply_decision_logic
from database.db_manager import init_db, insert_readings, insert_risk_results


def main():
    DATA_PATH    = "data/simulated/structural_data.csv"
    RESULTS_PATH = "data/simulated/risk_analysis_results.csv"

    # 0. Initialise database
    print("=" * 60)
    print("STRIDE — Structural Risk Intelligence Pipeline")
    print("=" * 60)
    print("\n[0/5] Initialising database...")
    init_db()

    if not os.path.exists(DATA_PATH):
        print("\n[1/5] Generating synthetic bridge structural data...")
        subprocess.run([sys.executable, "scripts/generate_synthetic_data.py"], check=True)
    else:
        print(f"\n[1/5] Data already exists at {DATA_PATH} — skipping generation.")

    # 2. Load
    print("\n[2/5] Loading structural data...")
    df = load_structural_data(DATA_PATH)
    print(f"      Loaded {len(df)} rows across {df['zone_id'].nunique()} zones.")

    # 3. Feature engineering
    print("\n[3/5] Computing structural features...")
    df = compute_features(df, data_source="simulated")

    # 4. Train + predict
    print("\n[4/5] Training risk model and generating predictions...")
    model = RiskModel()
    model.train(df)
    df = model.predict(df)
    model.save()   # Persist model.pkl for dashboard use

    # 5. Decision logic
    print("\n[5/5] Applying decision logic...")
    df = apply_decision_logic(df)

    # 6. Persist to database
    print("\n[6/6] Writing to database...")
    # Write raw sensor readings (original columns only)
    raw_cols = ['zone_id', 'timestamp', 'deformation_mm', 'gap_mm',
                'vibration_level', 'temperature', 'load_estimate']
    n_readings = insert_readings(df[raw_cols], data_source="simulated")
    n_results  = insert_risk_results(df, data_source="simulated")
    print(f"      Stored {n_readings} sensor readings, {n_results} risk snapshots.")

    # Save results
    os.makedirs("data/simulated", exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("STRUCTURAL RISK SUMMARY — LATEST STATUS PER ZONE")
    print("=" * 60)

    latest = df.groupby('zone_id').tail(1).sort_values('sri', ascending=False)
    for _, row in latest.iterrows():
        etc_str = f"{row['etc']:.1f} hrs" if row['etc'] not in (float('inf'), 0.0) else (
            "CRITICAL" if row['etc'] == 0.0 else "Stable"
        )
        print(f"\n  Zone : {row['zone_id']}")
        print(f"  SRI  : {row['sri']:.3f}  |  DPR: {row['dpr']:.5f}/hr  |  ETC: {etc_str}")
        print(f"  Risk : {row['urgency_category']}")
        print(f"  Note : {row['urgency_explanation']}")

    print(f"\nResults saved -> {RESULTS_PATH}")
    print("Run `streamlit run app.py` to launch the dashboard.\n")


if __name__ == "__main__":
    main()