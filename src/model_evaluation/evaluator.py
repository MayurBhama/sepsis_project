import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.exception import CustomException
from src.logger import logger


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    threshold: float


class ModelEvaluator:

    def __init__(self):
        pass

    def evaluate(
        self,
        y_true,
        y_proba,
        threshold: float = 0.5,
    ) -> EvaluationResult:
        """
        Compute metrics at a given classification threshold.
        """

        try:
            logger.info(f"Evaluating model at threshold={threshold:.2f}")

            # binary predictions based on threshold
            y_pred = (y_proba >= threshold).astype(int)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            try:
                auc = roc_auc_score(y_true, y_proba)
            except ValueError:
                auc = float("nan")

            cm = confusion_matrix(y_true, y_pred)

            logger.info(
                f"Metrics -> Acc: {acc:.4f}, Prec: {prec:.4f}, "
                f"Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
            )

            return EvaluationResult(
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1=f1,
                roc_auc=auc,
                confusion_matrix=cm,
                threshold=threshold,
            )

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------------------------
    # NEW: Threshold Search Method (Fixes Trainer Error)
    # -------------------------------------------------------------------------
    def threshold_search(self, y_true, y_proba):
        """
        Evaluate the model across multiple thresholds and return the best one.
        Priority = Max Recall, then Max F1-score.
        """

        try:
            logger.info("Running threshold search...")

            thresholds = np.arange(0.05, 0.95, 0.05)

            best_eval = None

            for t in thresholds:
                eval_res = self.evaluate(y_true, y_proba, threshold=t)

                if best_eval is None:
                    best_eval = eval_res
                else:
                    # PRIORITY 1 → Max Recall
                    if eval_res.recall > best_eval.recall:
                        best_eval = eval_res
                    # PRIORITY 2 → If recall same, pick higher F1
                    elif eval_res.recall == best_eval.recall and eval_res.f1 > best_eval.f1:
                        best_eval = eval_res

            logger.info(
                f"Best threshold found: {best_eval.threshold:.2f} "
                f"(Recall={best_eval.recall:.4f}, F1={best_eval.f1:.4f})"
            )

            return best_eval

        except Exception as e:
            raise CustomException(e, sys)
