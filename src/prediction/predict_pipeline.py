import os
import sys
import pandas as pd
import numpy as np

from src.utils import load_object
from src.exception import CustomException
from src.logger import logger

class SepsisPredictor:

    def __init__(self, model_path="models/best_sepsis_model.pkl"):
        try:
            logger.info("Loading model package...")
            package = load_object(model_path)

            self.model = package["model"]
            self.scaler = package["scaler"]
            self.feature_names = package["feature_names"]
            self.best_threshold = package["best_threshold"]

            logger.info(f"Model, scaler, and metadata loaded successfully. Best threshold: {self.best_threshold}")

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------
    def preprocess(self, input_dict: dict) -> np.ndarray:
        """
        Convert input dictionary → DataFrame → scale → return numpy array
        """
        try:
            df = pd.DataFrame([input_dict])

            # Ensure all expected features exist
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0   # default value for missing columns

            df = df[self.feature_names]
            df = df.fillna(0)

            # Scale
            scaled = self.scaler.transform(df)
            return scaled

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------
    def predict(self, input_dict: dict) -> dict:
        """
        Perform full prediction: probability + label (based on threshold)
        """
        try:
            X = self.preprocess(input_dict)
            probability = float(self.model.predict_proba(X)[0][1])
            label = int(probability >= self.best_threshold)

            return {
                "probability": probability,
                "predicted_label": label,
                "threshold_used": self.best_threshold
            }

        except Exception as e:
            raise CustomException(e, sys)