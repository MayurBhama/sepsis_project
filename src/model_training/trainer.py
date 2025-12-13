"""
Ensemble Model Training Module for Sepsis Prediction

Implements multi-model ensemble with:
- LightGBM (primary - fast, handles missing values natively)
- XGBoost (robust gradient boosting)
- CatBoost (handles categorical features well)

With class imbalance handling and threshold optimization.
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

# Tree-based models
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml, save_object
from src.model_evaluation.evaluator import ModelEvaluator


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""
    random_state: int = 42
    n_folds: int = 5
    optuna_trials: int = 30
    
    # Model weights for ensemble
    lgbm_weight: float = 0.4
    xgb_weight: float = 0.4
    catboost_weight: float = 0.2
    
    # Paths
    model_dir: str = "models"
    best_model_path: str = "models/best_sepsis_model.pkl"
    metadata_path: str = "models/model_metadata.json"
    
    # Data paths
    train_path: str = "data/processed/train_sepsis.csv"
    val_path: str = "data/processed/val_sepsis.csv"
    test_path: str = "data/processed/test_sepsis.csv"
    
    # MLflow
    tracking_uri: str = "mlruns"
    experiment_name: str = "sepsis_prediction_experiments"


class EnsembleModelTrainer:
    """
    Ensemble trainer combining LightGBM, XGBoost, and CatBoost.
    Handles severe class imbalance and optimizes for clinical utility.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            cfg = read_yaml(config_path)
            
            mt_cfg = cfg.get("model_training", {})
            split_cfg = cfg.get("data_split", {})
            mlflow_cfg = cfg.get("mlflow", {})
            
            self.target_col = cfg["data_validation"]["target_column"]
            self.patient_col = split_cfg.get("patient_column", "Patient_ID")
            
            self.config = EnsembleConfig(
                random_state=mt_cfg.get("random_state", 42),
                n_folds=mt_cfg.get("n_folds", 5),
                optuna_trials=mt_cfg.get("optuna_trials", 30),
                model_dir=mt_cfg.get("model_dir", "models"),
                best_model_path=os.path.join(mt_cfg.get("model_dir", "models"), 
                                              mt_cfg.get("best_model_filename", "best_sepsis_model.pkl")),
                metadata_path=os.path.join(mt_cfg.get("model_dir", "models"), 
                                           mt_cfg.get("metadata_filename", "model_metadata.json")),
                train_path=split_cfg.get("train_path", "data/processed/train_sepsis.csv"),
                val_path=split_cfg.get("val_path", "data/processed/val_sepsis.csv"),
                test_path=split_cfg.get("test_path", "data/processed/test_sepsis.csv"),
                tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns"),
                experiment_name=mlflow_cfg.get("experiment_name", "sepsis_prediction_experiments"),
            )
            
            self.scaler = StandardScaler()
            self.evaluator = ModelEvaluator()
            
            # MLflow setup
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            
            logger.info("EnsembleModelTrainer initialized with LightGBM + XGBoost + CatBoost")
            logger.info(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
            logger.info(f"CatBoost available: {CATBOOST_AVAILABLE}")
            
        except Exception as e:
            raise CustomException(e, sys)

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation, and test datasets."""
        try:
            train_df = pd.read_csv(self.config.train_path)
            
            # Load validation if exists, else create from train
            try:
                val_df = pd.read_csv(self.config.val_path)
            except FileNotFoundError:
                logger.warning("Validation set not found, splitting from train...")
                from sklearn.model_selection import train_test_split
                train_df, val_df = train_test_split(
                    train_df, test_size=0.15, 
                    stratify=train_df[self.target_col],
                    random_state=self.config.random_state
                )
            
            test_df = pd.read_csv(self.config.test_path)
            
            logger.info(f"Loaded - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
            return train_df, val_df, test_df
            
        except Exception as e:
            raise CustomException(e, sys)

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target from dataframe."""
        try:
            # Columns to drop
            drop_cols = ['Unnamed: 0', self.patient_col, 'Patient_ID', 'Hour', 'ICULOS']
            drop_cols = [c for c in drop_cols if c in df.columns]
            
            df_clean = df.drop(columns=drop_cols, errors='ignore')
            
            X = df_clean.drop(columns=[self.target_col])
            y = df_clean[self.target_col]
            
            feature_names = list(X.columns)
            
            return X.values, y.values, feature_names
            
        except Exception as e:
            raise CustomException(e, sys)

    def _calculate_class_weights(self, y: np.ndarray) -> float:
        """Calculate scale_pos_weight for imbalanced data."""
        n_negative = (y == 0).sum()
        n_positive = (y == 1).sum()
        return n_negative / max(n_positive, 1)

    def _get_lgbm_params(self, scale_pos_weight: float, trial: Optional[optuna.Trial] = None) -> Dict:
        """Get LightGBM parameters, optionally from Optuna trial."""
        if trial:
            return {
                'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 500),
                'max_depth': trial.suggest_int('lgbm_max_depth', 3, 12),
                'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 150),
                'subsample': trial.suggest_float('lgbm_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 10, 100),
                'reg_alpha': trial.suggest_float('lgbm_reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('lgbm_reg_lambda', 0.0, 1.0),
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.config.random_state,
                'n_jobs': -1,
                'verbose': -1,
            }
        else:
            return {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 64,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.config.random_state,
                'n_jobs': -1,
                'verbose': -1,
            }

    def _get_xgb_params(self, scale_pos_weight: float, trial: Optional[optuna.Trial] = None) -> Dict:
        """Get XGBoost parameters."""
        if trial:
            return {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('xgb_gamma', 0.0, 5.0),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'aucpr',
                'random_state': self.config.random_state,
                'n_jobs': -1,
            }
        else:
            return {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'aucpr',
                'random_state': self.config.random_state,
                'n_jobs': -1,
            }

    def _get_catboost_params(self, class_weights: List[float], trial: Optional[optuna.Trial] = None) -> Dict:
        """Get CatBoost parameters."""
        if trial:
            return {
                'iterations': trial.suggest_int('cat_iterations', 100, 500),
                'depth': trial.suggest_int('cat_depth', 3, 10),
                'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.2),
                'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1.0, 10.0),
                'class_weights': class_weights,
                'random_state': self.config.random_state,
                'verbose': False,
            }
        else:
            return {
                'iterations': 300,
                'depth': 8,
                'learning_rate': 0.05,
                'class_weights': class_weights,
                'random_state': self.config.random_state,
                'verbose': False,
            }

    def _train_single_model(
        self, 
        model_type: str, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        params: Dict
    ):
        """Train a single model."""
        try:
            if model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
                model = LGBMClassifier(**params)
            elif model_type == 'xgb':
                model = XGBClassifier(**params)
            elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                model = CatBoostClassifier(**params)
            else:
                raise ValueError(f"Model type {model_type} not available")
            
            model.fit(X_train, y_train)
            return model
            
        except Exception as e:
            raise CustomException(e, sys)

    def _find_optimal_threshold(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """Find optimal threshold for given metric."""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score

    def train(self, quick_test: bool = False) -> Dict[str, Any]:
        """
        Main training pipeline.
        
        Args:
            quick_test: If True, use only 10% of data for quick validation
            
        Returns:
            Dictionary with model package and metrics
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING ENSEMBLE MODEL TRAINING")
            logger.info("=" * 60)
            
            # Load data
            train_df, val_df, test_df = self._load_data()
            
            # Quick test mode - use subset
            if quick_test:
                train_df = train_df.sample(frac=0.1, random_state=42)
                val_df = val_df.sample(frac=0.1, random_state=42)
                test_df = test_df.sample(frac=0.1, random_state=42)
                logger.info("QUICK TEST MODE: Using 10% of data")
            
            # Prepare features
            X_train, y_train, feature_names = self._prepare_features(train_df)
            X_val, y_val, _ = self._prepare_features(val_df)
            X_test, y_test, _ = self._prepare_features(test_df)
            
            logger.info(f"Features: {len(feature_names)}")
            logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            # Handle missing values for non-tree models (scale)
            X_train = np.nan_to_num(X_train, nan=0)
            X_val = np.nan_to_num(X_val, nan=0)
            X_test = np.nan_to_num(X_test, nan=0)
            
            # Calculate class weights
            scale_pos_weight = self._calculate_class_weights(y_train)
            class_weights = [1.0, scale_pos_weight]
            logger.info(f"Class imbalance ratio: 1:{scale_pos_weight:.1f}")
            
            # MLflow run
            with mlflow.start_run(run_name="ensemble_training"):
                
                mlflow.log_param("n_features", len(feature_names))
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("class_imbalance", scale_pos_weight)
                
                models = {}
                val_probas = {}
                
                # Train LightGBM
                if LIGHTGBM_AVAILABLE:
                    logger.info("Training LightGBM...")
                    lgbm_params = self._get_lgbm_params(scale_pos_weight)
                    models['lgbm'] = self._train_single_model('lgbm', X_train, y_train, lgbm_params)
                    val_probas['lgbm'] = models['lgbm'].predict_proba(X_val)[:, 1]
                    lgbm_auprc = average_precision_score(y_val, val_probas['lgbm'])
                    logger.info(f"LightGBM Val AUPRC: {lgbm_auprc:.4f}")
                    mlflow.log_metric("lgbm_val_auprc", lgbm_auprc)
                
                # Train XGBoost
                logger.info("Training XGBoost...")
                xgb_params = self._get_xgb_params(scale_pos_weight)
                models['xgb'] = self._train_single_model('xgb', X_train, y_train, xgb_params)
                val_probas['xgb'] = models['xgb'].predict_proba(X_val)[:, 1]
                xgb_auprc = average_precision_score(y_val, val_probas['xgb'])
                logger.info(f"XGBoost Val AUPRC: {xgb_auprc:.4f}")
                mlflow.log_metric("xgb_val_auprc", xgb_auprc)
                
                # Train CatBoost
                if CATBOOST_AVAILABLE:
                    logger.info("Training CatBoost...")
                    cat_params = self._get_catboost_params(class_weights)
                    models['catboost'] = self._train_single_model('catboost', X_train, y_train, cat_params)
                    val_probas['catboost'] = models['catboost'].predict_proba(X_val)[:, 1]
                    cat_auprc = average_precision_score(y_val, val_probas['catboost'])
                    logger.info(f"CatBoost Val AUPRC: {cat_auprc:.4f}")
                    mlflow.log_metric("catboost_val_auprc", cat_auprc)
                
                # Ensemble predictions
                logger.info("Creating ensemble predictions...")
                ensemble_val_proba = np.zeros(len(X_val))
                ensemble_weights = []
                
                if 'lgbm' in val_probas:
                    ensemble_val_proba += self.config.lgbm_weight * val_probas['lgbm']
                    ensemble_weights.append(('lgbm', self.config.lgbm_weight))
                    
                if 'xgb' in val_probas:
                    ensemble_val_proba += self.config.xgb_weight * val_probas['xgb']
                    ensemble_weights.append(('xgb', self.config.xgb_weight))
                    
                if 'catboost' in val_probas:
                    ensemble_val_proba += self.config.catboost_weight * val_probas['catboost']
                    ensemble_weights.append(('catboost', self.config.catboost_weight))
                
                # Normalize if not all models present
                total_weight = sum(w for _, w in ensemble_weights)
                ensemble_val_proba /= total_weight
                
                # Ensemble validation metrics
                ensemble_auprc = average_precision_score(y_val, ensemble_val_proba)
                ensemble_auroc = roc_auc_score(y_val, ensemble_val_proba)
                logger.info(f"Ensemble Val AUPRC: {ensemble_auprc:.4f}")
                logger.info(f"Ensemble Val AUROC: {ensemble_auroc:.4f}")
                mlflow.log_metric("ensemble_val_auprc", ensemble_auprc)
                mlflow.log_metric("ensemble_val_auroc", ensemble_auroc)
                
                # Find optimal threshold
                best_threshold, best_f1 = self._find_optimal_threshold(y_val, ensemble_val_proba)
                logger.info(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
                mlflow.log_metric("best_threshold", best_threshold)
                mlflow.log_metric("best_f1_val", best_f1)
                
                # Test set evaluation
                logger.info("Evaluating on test set...")
                ensemble_test_proba = np.zeros(len(X_test))
                
                if 'lgbm' in models:
                    ensemble_test_proba += self.config.lgbm_weight * models['lgbm'].predict_proba(X_test)[:, 1]
                if 'xgb' in models:
                    ensemble_test_proba += self.config.xgb_weight * models['xgb'].predict_proba(X_test)[:, 1]
                if 'catboost' in models:
                    ensemble_test_proba += self.config.catboost_weight * models['catboost'].predict_proba(X_test)[:, 1]
                
                ensemble_test_proba /= total_weight
                
                # Test metrics
                test_auprc = average_precision_score(y_test, ensemble_test_proba)
                test_auroc = roc_auc_score(y_test, ensemble_test_proba)
                
                test_pred = (ensemble_test_proba >= best_threshold).astype(int)
                test_f1 = f1_score(y_test, test_pred)
                test_recall = recall_score(y_test, test_pred)
                test_precision = precision_score(y_test, test_pred)
                
                logger.info(f"Test AUPRC: {test_auprc:.4f}")
                logger.info(f"Test AUROC: {test_auroc:.4f}")
                logger.info(f"Test F1: {test_f1:.4f}")
                logger.info(f"Test Recall: {test_recall:.4f}")
                logger.info(f"Test Precision: {test_precision:.4f}")
                
                mlflow.log_metric("test_auprc", test_auprc)
                mlflow.log_metric("test_auroc", test_auroc)
                mlflow.log_metric("test_f1", test_f1)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("test_precision", test_precision)
                
                # Save model package
                os.makedirs(self.config.model_dir, exist_ok=True)
                
                model_package = {
                    'models': models,
                    'scaler': self.scaler,
                    'feature_names': feature_names,
                    'best_threshold': float(best_threshold),
                    'ensemble_weights': ensemble_weights,
                    'performance': {
                        'validation': {
                            'auprc': float(ensemble_auprc),
                            'auroc': float(ensemble_auroc),
                            'f1': float(best_f1),
                        },
                        'test': {
                            'auprc': float(test_auprc),
                            'auroc': float(test_auroc),
                            'f1': float(test_f1),
                            'recall': float(test_recall),
                            'precision': float(test_precision),
                        }
                    }
                }
                
                save_object(self.config.best_model_path, model_package)
                logger.info(f"Model saved to: {self.config.best_model_path}")
                
                # Save metadata
                with open(self.config.metadata_path, 'w') as f:
                    json.dump(model_package['performance'], f, indent=4)
                
                logger.info("=" * 60)
                logger.info("ENSEMBLE TRAINING COMPLETED SUCCESSFULLY")
                logger.info("=" * 60)
                
                return model_package
                
        except Exception as e:
            raise CustomException(e, sys)


# Backward compatibility
class ModelTrainer(EnsembleModelTrainer):
    """Alias for backward compatibility."""
    pass


if __name__ == "__main__":
    trainer = EnsembleModelTrainer()
    trainer.train()