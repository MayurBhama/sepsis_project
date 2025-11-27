import sys
import pandas as pd
import numpy as np
import json

from src.utils import load_object
from src.logger import logger
from src.model_evaluation.evaluator import ModelEvaluator


def evaluate_on_test():
    try:
        logger.info("===== LOADING MODEL PACKAGE =====")

        model_package = load_object("models/best_sepsis_model.pkl")
        model = model_package["model"]
        scaler = model_package["scaler"]
        feature_names = model_package["feature_names"]
        best_threshold = model_package["best_threshold"]

        logger.info(f"Loaded model with best threshold: {best_threshold}")

        # Load test data
        logger.info("===== LOADING TEST DATA =====")
        test_df = pd.read_csv("data/processed/test_sepsis.csv")

        X_test = test_df[feature_names]
        y_test = test_df["SepsisLabel"]

        # Scale
        X_test_scaled = scaler.transform(X_test)

        # Predict
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= best_threshold).astype(int)

        evaluator = ModelEvaluator()
        eval_result = evaluator.evaluate(y_test, y_proba, threshold=best_threshold)

        logger.info("===== FINAL TEST METRICS =====")
        logger.info(f"Accuracy  : {eval_result.accuracy:.4f}")
        logger.info(f"Precision : {eval_result.precision:.4f}")
        logger.info(f"Recall    : {eval_result.recall:.4f}")
        logger.info(f"F1-Score  : {eval_result.f1:.4f}")
        logger.info(f"ROC-AUC   : {eval_result.roc_auc:.4f}")
        logger.info(f"Threshold : {eval_result.threshold}")
        logger.info(f"Confusion Matrix:\n{eval_result.confusion_matrix}")

        print("===== FINAL TEST METRICS =====")
        print(f"Accuracy  : {eval_result.accuracy:.4f}")
        print(f"Precision : {eval_result.precision:.4f}")
        print(f"Recall    : {eval_result.recall:.4f}")
        print(f"F1-Score  : {eval_result.f1:.4f}")
        print(f"ROC-AUC   : {eval_result.roc_auc:.4f}")
        print(f"Confusion Matrix:\n{eval_result.confusion_matrix}")

    except Exception as e:
        print("ERROR:", e)
        raise e


if __name__ == "__main__":
    evaluate_on_test()
