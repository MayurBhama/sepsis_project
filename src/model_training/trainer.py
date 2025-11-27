import os
import sys
import json
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml, save_object
from src.model_evaluation.evaluator import ModelEvaluator, EvaluationResult


# -------------------------------------------------------------------------
# CONFIG DATACLASS
# -------------------------------------------------------------------------
@dataclass
class TrainerConfig:
    random_state: int
    smote_enabled: bool
    smote_k_neighbors: int
    model_dir: str
    best_model_path: str
    metadata_path: str
    mlflow_cfg: Dict[str, Any]
    train_path: str
    test_path: str


# -------------------------------------------------------------------------
# MAIN TRAINER CLASS
# -------------------------------------------------------------------------
class ModelTrainer:

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        try:
            cfg = read_yaml(config_path)

            mt_cfg = cfg["model_training"]
            split_cfg = cfg["data_split"]
            mlflow_cfg = cfg["mlflow"]

            self.target_col = cfg["data_validation"]["target_column"]

            self.trainer_config = TrainerConfig(
                random_state=mt_cfg["random_state"],
                smote_enabled=mt_cfg["smote"]["enabled"],
                smote_k_neighbors=mt_cfg["smote"]["k_neighbors"],
                model_dir=mt_cfg["model_dir"],
                best_model_path=os.path.join(mt_cfg["model_dir"], mt_cfg["best_model_filename"]),
                metadata_path=os.path.join(mt_cfg["model_dir"], mt_cfg["metadata_filename"]),
                mlflow_cfg=mlflow_cfg,
                train_path=split_cfg["train_path"],
                test_path=split_cfg["test_path"],
            )

            self.scaler = StandardScaler()
            self.evaluator = ModelEvaluator()

            # MLflow setup
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
            mlflow.set_experiment(mlflow_cfg["experiment_name"])

            logger.info("ModelTrainer initialized with clean feature handling + CV + Optuna")

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # LOAD TRAIN / TEST CSV FILES
    # ------------------------------------------------------------------
    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_df = pd.read_csv(self.trainer_config.train_path)
            test_df = pd.read_csv(self.trainer_config.test_path)

            logger.info(f"Loaded train: {train_df.shape} | Loaded test: {test_df.shape}")
            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # REMOVE UNUSED COLUMNS
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = ["Unnamed: 0", "Patient_ID"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        return df

    # ------------------------------------------------------------------
    # PREPARE TRAIN / VAL / TEST DATASETS
    # ------------------------------------------------------------------
    def _prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):

        train_df = self._clean_features(train_df)
        test_df = self._clean_features(test_df)

        # Split X/y
        X_train_full = train_df.drop(columns=[self.target_col])
        y_train_full = train_df[self.target_col]

        X_test = test_df.drop(columns=[self.target_col])
        y_test = test_df[self.target_col]

        # Inner validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            random_state=self.trainer_config.random_state,
            stratify=y_train_full
        )

        logger.info(f"Inner split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

        # Missing values handling
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)

        # Scaling
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # SMOTE only on train
        if self.trainer_config.smote_enabled:
            logger.info("Applying SMOTE...")
            smote = SMOTE(
                random_state=self.trainer_config.random_state,
                k_neighbors=self.trainer_config.smote_k_neighbors
            )
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        else:
            logger.info("SMOTE disabled.")

        feature_names = list(X_train_full.columns)

        return (
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test,
            feature_names
        )

    # ------------------------------------------------------------------
    # OPTUNA OBJECTIVE (PR-AUC)
    # ------------------------------------------------------------------
    def _objective(self, trial, X_train_scaled, y_train, X_val_scaled, y_val):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "eval_metric": "logloss",
            "random_state": self.trainer_config.random_state,
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)
        model.fit(X_train_scaled, y_train)

        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        pr_auc = average_precision_score(y_val, y_val_proba)

        # Log Optuna trial
        with mlflow.start_run(run_name="xgb_trial", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("pr_auc_val", pr_auc)

        return pr_auc

    # ------------------------------------------------------------------
    # CONVERT EVAL TO JSON
    # ------------------------------------------------------------------
    @staticmethod
    def _eval_to_dict(ev: EvaluationResult) -> Dict[str, Any]:
        return {
            "accuracy": float(ev.accuracy),
            "precision": float(ev.precision),
            "recall": float(ev.recall),
            "f1": float(ev.f1),
            "roc_auc": float(ev.roc_auc),
            "threshold": float(ev.threshold),
            "confusion_matrix": ev.confusion_matrix.tolist()
        }

    # ------------------------------------------------------------------
    # MAIN TRAINING PIPELINE
    # ------------------------------------------------------------------
    def train(self):

        try:
            # Load train & test CSVs
            train_df, test_df = self._load_data()

            # Prepare data
            (
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                X_test_scaled, y_test,
                feature_names
            ) = self._prepare_data(train_df, test_df)

            logger.info("Starting Optuna hyperparameter search...")

            with mlflow.start_run(run_name="optuna_xgb_training"):

                study = optuna.create_study(direction="maximize")
                study.optimize(
                    lambda trial: self._objective(
                        trial, X_train_scaled, y_train,
                        X_val_scaled, y_val
                    ),
                    n_trials=20,
                )

                mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
                mlflow.log_metric("best_pr_auc_val", study.best_value)

                # Train final model
                best_model = XGBClassifier(
                    **study.best_params,
                    eval_metric="logloss",
                    random_state=self.trainer_config.random_state,
                    n_jobs=-1,
                )
                best_model.fit(X_train_scaled, y_train)

                # Threshold search
                y_val_proba = best_model.predict_proba(X_val_scaled)[:, 1]
                best_val_eval = self.evaluator.threshold_search(y_val, y_val_proba)

                # Evaluate on test data
                y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                final_test_eval = self.evaluator.evaluate(
                    y_test, y_test_proba, threshold=best_val_eval.threshold
                )

                # Save model package
                os.makedirs(self.trainer_config.model_dir, exist_ok=True)

                model_package = {
                    "model": best_model,
                    "scaler": self.scaler,
                    "feature_names": feature_names,
                    "best_threshold": float(best_val_eval.threshold),
                    "performance": {
                        "validation": self._eval_to_dict(best_val_eval),
                        "test": self._eval_to_dict(final_test_eval),
                    },
                }

                save_object(self.trainer_config.best_model_path, model_package)

                # Save metadata
                with open(self.trainer_config.metadata_path, "w") as f:
                    json.dump(model_package["performance"], f, indent=4)

                logger.info("===== TRAINING COMPLETED SUCCESSFULLY =====")

        except Exception as e:
            raise CustomException(e, sys)