"""
Patient-Stratified Data Splitting Module

CRITICAL: This module ensures NO patient data leakage between train/val/test sets.
Each patient's entire ICU stay belongs to exactly ONE split.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.utils import read_yaml
from src.logger import logger
from src.exception import CustomException
import pymongo


class PatientStratifiedSplitter:
    """
    Splits data ensuring patient-level integrity:
    - No patient appears in multiple splits
    - Stratified by sepsis outcome at patient level
    - Produces train/val/test splits (70/15/15 or configurable)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            self.cfg = read_yaml(config_path)
            self.mongo_cfg = self.cfg["mongodb"]
            self.split_cfg = self.cfg["data_split"]
            self.target_col = self.cfg["data_validation"]["target_column"]
            
            # Patient ID column
            self.patient_col = self.split_cfg.get("patient_column", "Patient_ID")
            
            # Split ratios
            self.test_size = self.split_cfg.get("test_size", 0.15)
            self.val_size = self.split_cfg.get("val_size", 0.15)
            self.random_state = self.split_cfg.get("random_state", 42)
            
            logger.info("PatientStratifiedSplitter initialized successfully.")
            
        except Exception as e:
            raise CustomException(e, sys)

    def _load_data(self) -> pd.DataFrame:
        """Load cleaned data from MongoDB or CSV."""
        try:
            # Try MongoDB first
            client = pymongo.MongoClient(self.mongo_cfg["uri"])
            db = client[self.mongo_cfg["database"]]
            collection = db[self.mongo_cfg["clean_collection"]]
            
            data = list(collection.find({}, {"_id": 0}))
            
            if data:
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df):,} records from MongoDB")
            else:
                # Fallback to CSV
                csv_path = self.cfg["preprocessing"]["output_path"]
                df = pd.read_csv(csv_path)
                logger.info(f"Loaded {len(df):,} records from CSV: {csv_path}")
            
            return df
            
        except Exception as e:
            # Final fallback - raw CSV
            logger.warning(f"MongoDB load failed: {e}. Trying raw CSV...")
            csv_path = self.cfg["data_source"]["csv_path"]
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df):,} records from raw CSV: {csv_path}")
            return df

    def _get_patient_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get patient-level sepsis labels.
        A patient is labeled as sepsis=1 if they developed sepsis at any point.
        """
        patient_labels = df.groupby(self.patient_col)[self.target_col].max().reset_index()
        patient_labels.columns = [self.patient_col, 'patient_sepsis']
        return patient_labels

    def _stratified_patient_split(
        self, 
        patient_labels: pd.DataFrame,
        test_size: float,
        random_state: int
    ) -> tuple:
        """
        Split patients into two groups with stratification by sepsis label.
        Uses manual stratified sampling to handle severe imbalance.
        """
        np.random.seed(random_state)
        
        # Separate sepsis and non-sepsis patients
        sepsis_patients = patient_labels[patient_labels['patient_sepsis'] == 1][self.patient_col].values
        non_sepsis_patients = patient_labels[patient_labels['patient_sepsis'] == 0][self.patient_col].values
        
        # Shuffle
        np.random.shuffle(sepsis_patients)
        np.random.shuffle(non_sepsis_patients)
        
        # Calculate split sizes
        n_sepsis_test = max(1, int(len(sepsis_patients) * test_size))
        n_non_sepsis_test = int(len(non_sepsis_patients) * test_size)
        
        # Split
        sepsis_test = sepsis_patients[:n_sepsis_test]
        sepsis_train = sepsis_patients[n_sepsis_test:]
        
        non_sepsis_test = non_sepsis_patients[:n_non_sepsis_test]
        non_sepsis_train = non_sepsis_patients[n_non_sepsis_test:]
        
        # Combine
        test_patients = np.concatenate([sepsis_test, non_sepsis_test])
        train_patients = np.concatenate([sepsis_train, non_sepsis_train])
        
        return train_patients, test_patients

    def split(self) -> tuple:
        """
        Main split function.
        Returns: (train_df, val_df, test_df)
        """
        try:
            # Load data
            df = self._load_data()
            
            # Validate patient column exists
            if self.patient_col not in df.columns:
                # Try to find alternative patient ID column
                alt_cols = ['Unnamed: 0', 'patient_id', 'PatientID', 'id']
                for col in alt_cols:
                    if col in df.columns:
                        self.patient_col = col
                        logger.info(f"Using '{col}' as patient identifier")
                        break
                else:
                    raise ValueError(f"No patient identifier found. Tried: {[self.patient_col] + alt_cols}")
            
            # Get patient-level labels
            patient_labels = self._get_patient_labels(df)
            n_patients = len(patient_labels)
            n_sepsis = (patient_labels['patient_sepsis'] == 1).sum()
            
            logger.info(f"Total patients: {n_patients:,}")
            logger.info(f"Patients with sepsis: {n_sepsis:,} ({n_sepsis/n_patients*100:.1f}%)")
            
            # First split: train+val vs test
            combined_test_size = self.test_size + self.val_size
            trainval_patients, test_patients = self._stratified_patient_split(
                patient_labels, 
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            # Second split: train vs val (from trainval)
            trainval_labels = patient_labels[patient_labels[self.patient_col].isin(trainval_patients)]
            relative_val_size = self.val_size / (1 - self.test_size)
            
            train_patients, val_patients = self._stratified_patient_split(
                trainval_labels,
                test_size=relative_val_size,
                random_state=self.random_state + 1
            )
            
            # Create dataframes
            train_df = df[df[self.patient_col].isin(train_patients)].copy()
            val_df = df[df[self.patient_col].isin(val_patients)].copy()
            test_df = df[df[self.patient_col].isin(test_patients)].copy()
            
            # Log statistics
            logger.info("=" * 60)
            logger.info("PATIENT-STRATIFIED SPLIT RESULTS")
            logger.info("=" * 60)
            logger.info(f"Train: {len(train_patients):,} patients, {len(train_df):,} records")
            logger.info(f"Val:   {len(val_patients):,} patients, {len(val_df):,} records")
            logger.info(f"Test:  {len(test_patients):,} patients, {len(test_df):,} records")
            
            # Verify no overlap
            train_set = set(train_patients)
            val_set = set(val_patients)
            test_set = set(test_patients)
            
            assert len(train_set & val_set) == 0, "LEAK: Train-Val overlap!"
            assert len(train_set & test_set) == 0, "LEAK: Train-Test overlap!"
            assert len(val_set & test_set) == 0, "LEAK: Val-Test overlap!"
            
            logger.info("✅ VERIFIED: No patient overlap between splits")
            
            # Log sepsis distribution per split
            train_sepsis = train_df[self.target_col].mean() * 100
            val_sepsis = val_df[self.target_col].mean() * 100
            test_sepsis = test_df[self.target_col].mean() * 100
            
            logger.info(f"Sepsis rate - Train: {train_sepsis:.2f}%, Val: {val_sepsis:.2f}%, Test: {test_sepsis:.2f}%")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            raise CustomException(e, sys)

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save split datasets to CSV."""
        try:
            # Create output directory
            os.makedirs("data/processed", exist_ok=True)
            
            # Save paths
            train_path = self.split_cfg.get("train_path", "data/processed/train_sepsis.csv")
            val_path = self.split_cfg.get("val_path", "data/processed/val_sepsis.csv")
            test_path = self.split_cfg.get("test_path", "data/processed/test_sepsis.csv")
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Saved train: {train_path} -> {train_df.shape}")
            logger.info(f"Saved val:   {val_path} -> {val_df.shape}")
            logger.info(f"Saved test:  {test_path} -> {test_df.shape}")
            logger.info("===== DATA SPLIT COMPLETED SUCCESSFULLY =====")
            
        except Exception as e:
            raise CustomException(e, sys)


def run_split_pipeline(config_path: str = "config/config.yaml"):
    """Main entry point for splitting pipeline."""
    try:
        splitter = PatientStratifiedSplitter(config_path)
        train_df, val_df, test_df = splitter.split()
        splitter.save_splits(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
        
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_split_pipeline()