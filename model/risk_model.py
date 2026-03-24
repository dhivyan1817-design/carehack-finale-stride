import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


class RiskModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.feature_cols = [
            'deformation_mm',
            'gap_mm',
            'vibration_level',
            'load_estimate',
            'deformation_rate',
            'damage_acceleration',
            'gap_growth_rate',
            'vibration_sensitivity',
        ]
        self._trained = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path=MODEL_PATH):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load(self, path=MODEL_PATH):
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self._trained = True
        print(f"Model loaded from {path}")
        return True

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _generate_labels(self, df):
        """
        Physics-inspired heuristic risk score for supervised training.
        Score increases with deformation, growth rate, and acceleration.
        """
        score = (
            0.25 * (df['deformation_mm'] / 60.0).clip(0, 1) +
            0.35 * (df['deformation_rate'] * 4.0).clip(0, 1) +
            0.25 * (df['damage_acceleration'] * 8.0).clip(0, 1) +
            0.15 * (df['vibration_sensitivity'] * 2.0).clip(0, 1)
        )
        return np.clip(score, 0, 1)

    def train(self, df):
        X = df[self.feature_cols].copy()
        y = self._generate_labels(df)
        self.model.fit(X, y)
        self._trained = True
        print("Model trained successfully.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, df):
        if not self._trained:
            raise RuntimeError("Model must be trained before calling predict().")

        df = df.copy()
        X = df[self.feature_cols]
        df['sri'] = np.clip(self.model.predict(X), 0, 1)

        # DPR: delta SRI / delta t  (per zone)
        df['dpr'] = (
            df.groupby('zone_id')['sri'].diff()
            / (df['dt'] + 1e-6)
        ).fillna(0)

        # ETC: hours until SRI reaches 0.8
        def calc_etc(row):
            if row['sri'] >= 0.8:
                return 0.0
            if row['dpr'] > 0:
                return min((0.8 - row['sri']) / row['dpr'], 9999.0)
            return np.inf  # stable or improving

        df['etc'] = df.apply(calc_etc, axis=1)
        return df
