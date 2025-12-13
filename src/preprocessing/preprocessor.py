"""
Advanced Data Preprocessing Module for Sepsis Prediction

Implements intelligent missing value handling for ICU time-series data:
- Forward-fill within patient (last known value)
- Backward-fill for remaining gaps
- Missingness indicators as features
- Patient-level median for labs
- Global median as final fallback
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml
from pymongo import MongoClient


class AdvancedPreprocessor:
    """
    Advanced preprocessing for ICU time-series data.
    Handles extreme missing data patterns while preserving temporal information.
    """

    # Feature groups for different imputation strategies
    VITAL_COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
    
    LAB_COLS = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 
                'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 
                'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
                'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets',
                'EtCO2', 'Bilirubin_direct', 'TroponinI', 'Fibrinogen']
    
    DEMO_COLS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
    
    # Features to drop (>99% missing, essentially useless)
    DROP_HIGH_MISSING = ['Bilirubin_direct', 'Fibrinogen', 'TroponinI']

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            self.validation_cfg = self.config["data_validation"]
            self.preproc_cfg = self.config["preprocessing"]
            
            self.target_col = self.validation_cfg["target_column"]
            self.patient_col = self.config["data_split"].get("patient_column", "Patient_ID")
            
            self.lower_pct = self.preproc_cfg["outlier"]["lower_percentile"]
            self.upper_pct = self.preproc_cfg["outlier"]["upper_percentile"]
            self.output_path = self.preproc_cfg["output_path"]
            
            # Imputation config
            self.impute_cfg = self.preproc_cfg.get("imputation", {})
            self.drop_high_missing = self.impute_cfg.get("drop_high_missing", True)
            self.create_missing_flags = self.impute_cfg.get("create_missing_flags", True)
            
            logger.info("AdvancedPreprocessor initialized successfully.")
            
        except Exception as e:
            raise CustomException(e, sys)

    def _get_available_cols(self, df: pd.DataFrame, col_list: List[str]) -> List[str]:
        """Return only columns that exist in the dataframe."""
        return [c for c in col_list if c in df.columns]

    # =========================================================================
    # DROP HIGH MISSING FEATURES
    # =========================================================================
    def _drop_useless_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop features with >99% missing values (essentially useless)."""
        try:
            if not self.drop_high_missing:
                return df
            
            logger.info("Dropping features with >99% missing values...")
            
            cols_to_drop = []
            for col in df.columns:
                if col in [self.target_col, self.patient_col, 'ICULOS', 'Hour']:
                    continue
                missing_pct = df[col].isna().sum() / len(df) * 100
                if missing_pct > 99:
                    cols_to_drop.append(col)
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # CREATE MISSINGNESS FLAGS
    # =========================================================================
    def _create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary flags indicating whether a value was missing.
        This is a powerful signal - missing lab = not ordered = less concern.
        """
        try:
            if not self.create_missing_flags:
                return df
            
            logger.info("Creating missingness indicator flags...")
            
            # Focus on key clinical features
            key_cols = self._get_available_cols(df, 
                ['Lactate', 'WBC', 'Creatinine', 'Platelets', 'BUN', 'Glucose', 
                 'pH', 'PaCO2', 'Temp', 'SBP', 'MAP', 'HR', 'Resp', 'O2Sat']
            )
            
            new_cols = 0
            for col in key_cols:
                flag_col = f'{col}_was_missing'
                df[flag_col] = df[col].isna().astype(int)
                new_cols += 1
            
            logger.info(f"Created {new_cols} missingness indicator columns")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # FORWARD/BACKWARD FILL (per patient)
    # =========================================================================
    def _forward_backward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using forward-fill then backward-fill within each patient.
        This preserves the temporal nature of ICU data.
        """
        try:
            logger.info("Applying forward-fill and backward-fill per patient...")
            
            # Columns to fill (vitals and labs)
            fill_cols = self._get_available_cols(df, self.VITAL_COLS + self.LAB_COLS)
            
            # Sort by patient and time
            time_col = 'ICULOS' if 'ICULOS' in df.columns else 'Hour'
            if self.patient_col in df.columns and time_col in df.columns:
                df = df.sort_values([self.patient_col, time_col])
            
            # Count missing before
            missing_before = df[fill_cols].isna().sum().sum()
            
            # Forward fill within patient groups
            if self.patient_col in df.columns:
                for col in fill_cols:
                    df[col] = df.groupby(self.patient_col)[col].ffill()
                    df[col] = df.groupby(self.patient_col)[col].bfill()
            else:
                # No patient ID - just ffill/bfill globally
                df[fill_cols] = df[fill_cols].ffill().bfill()
            
            # Count missing after
            missing_after = df[fill_cols].isna().sum().sum()
            filled = missing_before - missing_after
            
            logger.info(f"Forward/backward fill: reduced missing from {missing_before:,} to {missing_after:,} ({filled:,} filled)")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # GROUP-LEVEL MEDIAN IMPUTATION
    # =========================================================================
    def _group_median_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For remaining NaN values, use patient-level median first,
        then fall back to global median.
        """
        try:
            logger.info("Applying group-level median imputation for remaining NaNs...")
            
            fill_cols = self._get_available_cols(df, self.VITAL_COLS + self.LAB_COLS)
            
            missing_before = df[fill_cols].isna().sum().sum()
            
            if self.patient_col in df.columns:
                # Patient-level median
                for col in fill_cols:
                    # Fill with patient median
                    patient_medians = df.groupby(self.patient_col)[col].transform('median')
                    df[col] = df[col].fillna(patient_medians)
            
            # Global median for any remaining (new patients or all NaN patients)
            for col in fill_cols:
                if df[col].isna().any():
                    global_median = df[col].median()
                    if pd.notna(global_median):
                        df[col] = df[col].fillna(global_median)
                    else:
                        # Absolute fallback to 0 (rare)
                        df[col] = df[col].fillna(0)
            
            missing_after = df[fill_cols].isna().sum().sum()
            logger.info(f"Group median imputation: {missing_before:,} -> {missing_after:,}")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # DEMOGRAPHICS IMPUTATION
    # =========================================================================
    def _impute_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute demographic columns (should have minimal missing)."""
        try:
            logger.info("Imputing demographic columns...")
            
            demo_cols = self._get_available_cols(df, self.DEMO_COLS)
            
            for col in demo_cols:
                if df[col].isna().any():
                    if col == 'Age':
                        # Use median age
                        df[col] = df[col].fillna(df[col].median())
                    elif col == 'Gender':
                        # Use mode
                        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)
                    else:
                        # Unit flags - fill with 0
                        df[col] = df[col].fillna(0)
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # OUTLIER HANDLING
    # =========================================================================
    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers at configured percentiles."""
        try:
            logger.info(f"Capping outliers at [{self.lower_pct}, {self.upper_pct}] percentiles...")
            
            # Only cap clinical features, not engineered features
            cap_cols = self._get_available_cols(df, self.VITAL_COLS + self.LAB_COLS)
            
            for col in cap_cols:
                if col == self.target_col:
                    continue
                    
                lower_cap = df[col].quantile(self.lower_pct / 100)
                upper_cap = df[col].quantile(self.upper_pct / 100)
                
                outliers = ((df[col] < lower_cap) | (df[col] > upper_cap)).sum()
                if outliers > 0:
                    df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
            
            logger.info("Outlier capping complete")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # REMOVE DUPLICATES
    # =========================================================================
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        try:
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            
            if before > after:
                logger.info(f"Removed {before - after} duplicate rows")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # MAIN PREPROCESSING PIPELINE
    # =========================================================================
    def run_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline with intelligent missing value handling.
        
        Order of operations:
        1. Drop useless high-missing features
        2. Create missingness indicators (before filling!)
        3. Forward/backward fill per patient
        4. Group median imputation for remaining
        5. Demographics imputation
        6. Remove duplicates
        7. Cap outliers
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING ADVANCED PREPROCESSING PIPELINE")
            logger.info("=" * 60)
            logger.info(f"Input shape: {df.shape}")
            
            # 1. Drop features with >99% missing
            df = self._drop_useless_features(df)
            
            # 2. Create missingness indicators BEFORE filling
            df = self._create_missing_indicators(df)
            
            # 3. Forward/backward fill within patients
            df = self._forward_backward_fill(df)
            
            # 4. Group median for remaining NaN
            df = self._group_median_impute(df)
            
            # 5. Demographics
            df = self._impute_demographics(df)
            
            # 6. Remove duplicates
            df = self._remove_duplicates(df)
            
            # 7. Cap outliers
            df = self._cap_outliers(df)
            
            # Final check
            remaining_nan = df.isna().sum().sum()
            logger.info("=" * 60)
            logger.info(f"PREPROCESSING COMPLETE")
            logger.info(f"Final shape: {df.shape}")
            logger.info(f"Remaining NaN values: {remaining_nan}")
            logger.info("=" * 60)
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def save_preprocessed(self, df: pd.DataFrame) -> None:
        """Save cleaned data to CSV and optionally MongoDB."""
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            df.to_csv(self.output_path, index=False)
            logger.info(f"Preprocessed data saved to: {self.output_path}")
            
            # Optional: Save to MongoDB
            try:
                mongo_cfg = self.config["mongodb"]
                client = MongoClient(mongo_cfg["uri"])
                db = client[mongo_cfg["database"]]
                collection = db[mongo_cfg["clean_collection"]]
                
                collection.delete_many({})
                
                # Insert in batches to avoid memory issues
                batch_size = 10000
                records = df.to_dict("records")
                for i in range(0, len(records), batch_size):
                    batch = records[i:i+batch_size]
                    collection.insert_many(batch)
                
                logger.info(f"Saved to MongoDB: {mongo_cfg['database']}.{mongo_cfg['clean_collection']}")
                
            except Exception as mongo_error:
                logger.warning(f"MongoDB save skipped: {mongo_error}")
            
        except Exception as e:
            raise CustomException(e, sys)


# Backward compatibility
class DataPreprocessor(AdvancedPreprocessor):
    """Alias for backward compatibility."""
    pass
