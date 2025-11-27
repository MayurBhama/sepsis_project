# app/predictor.py

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.utils import read_yaml
from src.logger import logger
from src.exception import CustomException

import pickle


class SepsisPredictor:
    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            cfg = read_yaml(config_path)
            mt_cfg = cfg["model_training"]

            model_path = os.path.join(
                mt_cfg["model_dir"],
                mt_cfg["best_model_filename"]
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            logger.info(f"Loading model package from: {model_path}")

            with open(model_path, "rb") as f:
                model_package = pickle.load(f)

            # unpack
            self.model = model_package["model"]
            self.scaler = model_package["scaler"]
            self.feature_names = model_package["feature_names"]

            # PRINT FEATURE NAMES ON STARTUP (Your Request)
            print("\n===============================")
            print(" MODEL EXPECTED FEATURES")
            print("===============================")
            for i, feat in enumerate(self.feature_names, 1):
                print(f"{i}. {feat}")
            print("===============================\n")

            # Resolve threshold
            if "best_threshold" in model_package:
                self.threshold = float(model_package["best_threshold"])
            elif "performance" in model_package and "threshold" in model_package["performance"]:
                self.threshold = float(model_package["performance"]["threshold"])
            else:
                self.threshold = 0.5   # fallback

            logger.info(
                f"SepsisPredictor initialized. "
                f"Features: {len(self.feature_names)}, Threshold: {self.threshold}"
            )

            # ensure logs dir
            os.makedirs("logs", exist_ok=True)
            self.prediction_log_path = os.path.join("logs", "predictions.log")

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------------------------------
    def _build_feature_dataframe(self, payload: dict) -> pd.DataFrame:
        """
        Convert incoming JSON payload into a DataFrame aligned
        with model.feature_names.
        """
        try:
            df = pd.DataFrame([payload])

            # Add missing cols with default = 0
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0

            df = df[self.feature_names]  # exact order needed
            df = df.fillna(0.0)

            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------------------------------------
    def _log_prediction(self, payload: dict, probability: float, label: int):
        """
        Log prediction details to file.
        """
        try:
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "probability": float(probability),
                "predicted_label": int(label),
                "threshold_used": float(self.threshold),
                "input": payload,
            }

            with open(self.prediction_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.info(
                f"Prediction logged: prob={probability:.4f}, label={label}, "
                f"threshold={self.threshold}"
            )

        except Exception as e:
            logger.error(f"Prediction logging failed: {e}")

    # -----------------------------------------------------------
    def predict(self, payload: dict) -> dict:
        """
        Main prediction logic.
        """
        try:
            df = self._build_feature_dataframe(payload)
            X_scaled = self.scaler.transform(df)

            proba = float(self.model.predict_proba(X_scaled)[0, 1])
            label = int(proba >= self.threshold)

            self._log_prediction(payload, proba, label)

            return {
                "probability": round(proba, 4),
                "predicted_label": label,
                "threshold_used": float(self.threshold),
            }

        except Exception as e:
            raise CustomException(e, sys)


# Global instance for FastAPI
predictor = SepsisPredictor()
