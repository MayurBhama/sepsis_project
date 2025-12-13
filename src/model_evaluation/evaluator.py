"""
Clinical Evaluation Module for Sepsis Prediction

Implements clinical-focused metrics:
- AUPRC (Area Under Precision-Recall Curve) - critical for imbalanced data
- Sensitivity at specific specificities
- PhysioNet utility score
- Threshold optimization
"""

import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.exception import CustomException
from src.logger import logger


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    auprc: float  # NEW: Critical for imbalanced data
    confusion_matrix: np.ndarray
    threshold: float
    
    # Optional additional metrics
    sensitivity_at_95_spec: Optional[float] = None
    specificity_at_95_sens: Optional[float] = None


@dataclass
class ClinicalMetrics:
    """Clinical-focused metrics for sepsis prediction."""
    auprc: float  # Primary metric for imbalanced data
    auroc: float
    f1: float
    recall: float  # Sensitivity
    precision: float  # PPV
    specificity: float
    npv: float  # Negative Predictive Value
    
    # Operating point metrics
    sensitivity_at_95_spec: float
    sensitivity_at_90_spec: float
    
    # Utility score
    utility_score: Optional[float] = None


class ModelEvaluator:
    """
    Comprehensive model evaluator with clinical metrics
    optimized for sepsis prediction.
    """

    def __init__(self):
        logger.info("ModelEvaluator initialized with clinical metrics support")

    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> EvaluationResult:
        """
        Compute metrics at a given classification threshold.
        """
        try:
            y_true = np.asarray(y_true)
            y_proba = np.asarray(y_proba)
            
            # Binary predictions
            y_pred = (y_proba >= threshold).astype(int)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            try:
                auc = roc_auc_score(y_true, y_proba)
            except ValueError:
                auc = 0.5

            try:
                auprc = average_precision_score(y_true, y_proba)
            except ValueError:
                auprc = 0.0

            cm = confusion_matrix(y_true, y_pred)
            
            # Sensitivity at 95% specificity
            sens_at_95_spec = self._sensitivity_at_specificity(y_true, y_proba, 0.95)

            return EvaluationResult(
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1=f1,
                roc_auc=auc,
                auprc=auprc,
                confusion_matrix=cm,
                threshold=threshold,
                sensitivity_at_95_spec=sens_at_95_spec,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def _sensitivity_at_specificity(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        target_specificity: float
    ) -> float:
        """Find sensitivity at a given specificity level."""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            specificity = 1 - fpr
            
            # Find index where specificity >= target
            valid_indices = np.where(specificity >= target_specificity)[0]
            
            if len(valid_indices) == 0:
                return 0.0
            
            # Get the sensitivity at the lowest valid threshold
            idx = valid_indices[0]
            return float(tpr[idx])
            
        except Exception:
            return 0.0

    def compute_clinical_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5
    ) -> ClinicalMetrics:
        """
        Compute comprehensive clinical metrics.
        """
        try:
            y_true = np.asarray(y_true)
            y_proba = np.asarray(y_proba)
            y_pred = (y_proba >= threshold).astype(int)
            
            # Basic metrics
            auprc = average_precision_score(y_true, y_proba)
            auroc = roc_auc_score(y_true, y_proba)
            
            # Confusion matrix components
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            recall = tp / max(tp + fn, 1)  # Sensitivity
            precision = tp / max(tp + fp, 1)  # PPV
            specificity = tn / max(tn + fp, 1)
            npv = tn / max(tn + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            
            # Sensitivity at operating points
            sens_at_95_spec = self._sensitivity_at_specificity(y_true, y_proba, 0.95)
            sens_at_90_spec = self._sensitivity_at_specificity(y_true, y_proba, 0.90)
            
            return ClinicalMetrics(
                auprc=auprc,
                auroc=auroc,
                f1=f1,
                recall=recall,
                precision=precision,
                specificity=specificity,
                npv=npv,
                sensitivity_at_95_spec=sens_at_95_spec,
                sensitivity_at_90_spec=sens_at_90_spec,
            )
            
        except Exception as e:
            raise CustomException(e, sys)

    def compute_utility_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        hours_before_sepsis: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute PhysioNet 2019 Challenge utility score.
        
        Scoring:
        - True Positive (6h before sepsis): +1
        - True Positive (too early): decreasing reward
        - False Positive: -0.05
        - False Negative: -2.0
        - True Negative: 0
        """
        try:
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            
            # Simplified utility without time information
            if hours_before_sepsis is None:
                tp = ((y_true == 1) & (y_pred == 1)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                fn = ((y_true == 1) & (y_pred == 0)).sum()
                
                utility = 1.0 * tp - 0.05 * fp - 2.0 * fn
                
                # Normalize by number of positive cases
                n_positive = max(y_true.sum(), 1)
                normalized_utility = utility / n_positive
                
                return float(normalized_utility)
            
            # Full utility with time weighting
            utility = 0.0
            for i in range(len(y_true)):
                if y_true[i] == 1 and y_pred[i] == 1:
                    # True positive - reward based on timing
                    hours = hours_before_sepsis[i] if hours_before_sepsis is not None else 6
                    if 0 <= hours <= 12:
                        utility += 1.0
                    elif hours > 12:
                        utility += max(0, 1.0 - (hours - 12) / 12)
                    else:
                        utility += 0.5  # Late prediction
                        
                elif y_true[i] == 0 and y_pred[i] == 1:
                    utility -= 0.05  # False positive
                    
                elif y_true[i] == 1 and y_pred[i] == 0:
                    utility -= 2.0  # False negative (missed sepsis)
            
            return float(utility)
            
        except Exception as e:
            raise CustomException(e, sys)

    def threshold_search(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        metric: str = 'f1',
        min_recall: float = 0.3
    ) -> EvaluationResult:
        """
        Search for optimal threshold based on specified metric.
        
        Args:
            metric: 'f1', 'recall', or 'auprc'
            min_recall: Minimum recall constraint
        """
        try:
            logger.info(f"Searching for optimal threshold (metric={metric}, min_recall={min_recall})")

            thresholds = np.arange(0.05, 0.9, 0.02)
            best_eval = None
            best_score = -1

            for t in thresholds:
                eval_res = self.evaluate(y_true, y_proba, threshold=t)
                
                # Skip if recall is below minimum
                if eval_res.recall < min_recall:
                    continue
                
                # Get score for optimization
                if metric == 'f1':
                    score = eval_res.f1
                elif metric == 'recall':
                    score = eval_res.recall
                elif metric == 'auprc':
                    score = eval_res.auprc
                else:
                    score = eval_res.f1
                
                if score > best_score:
                    best_score = score
                    best_eval = eval_res

            if best_eval is None:
                # Fallback to default threshold
                best_eval = self.evaluate(y_true, y_proba, threshold=0.3)
                logger.warning("No threshold met constraints, using 0.3")
            else:
                logger.info(
                    f"Optimal threshold: {best_eval.threshold:.2f} "
                    f"(F1={best_eval.f1:.4f}, Recall={best_eval.recall:.4f}, AUPRC={best_eval.auprc:.4f})"
                )

            return best_eval

        except Exception as e:
            raise CustomException(e, sys)

    def generate_report(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5
    ) -> str:
        """Generate a text report of all metrics."""
        try:
            eval_res = self.evaluate(y_true, y_proba, threshold)
            clinical = self.compute_clinical_metrics(y_true, y_proba, threshold)
            
            report = f"""
{'='*60}
SEPSIS PREDICTION MODEL EVALUATION REPORT
{'='*60}

THRESHOLD: {threshold:.2f}

PRIMARY METRICS (for imbalanced data):
  AUPRC:              {clinical.auprc:.4f}
  AUROC:              {clinical.auroc:.4f}

CLASSIFICATION METRICS:
  Accuracy:           {eval_res.accuracy:.4f}
  Precision (PPV):    {clinical.precision:.4f}
  Recall (Sensitivity): {clinical.recall:.4f}
  Specificity:        {clinical.specificity:.4f}
  F1-Score:           {clinical.f1:.4f}
  NPV:                {clinical.npv:.4f}

CLINICAL OPERATING POINTS:
  Sensitivity @ 95% Specificity: {clinical.sensitivity_at_95_spec:.4f}
  Sensitivity @ 90% Specificity: {clinical.sensitivity_at_90_spec:.4f}

CONFUSION MATRIX:
{eval_res.confusion_matrix}

{'='*60}
"""
            return report
            
        except Exception as e:
            raise CustomException(e, sys)
