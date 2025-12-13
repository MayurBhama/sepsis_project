"""
Advanced Feature Engineering Module for Sepsis Prediction

Creates 30+ features critical for temporal ICU data:
- Lag features (previous values)
- Delta features (rate of change)
- Rolling statistics (trends)
- Clinical scores (Shock Index, qSOFA)
- Missingness indicators (clinical concern proxy)
- Time-based features
"""

import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for ICU time-series data.
    All features are computed per-patient to maintain temporal integrity.
    """

    # Define feature groups
    VITAL_COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
    
    KEY_LAB_COLS = ['Lactate', 'WBC', 'Creatinine', 'BUN', 'Platelets', 
                   'Glucose', 'pH', 'PaCO2', 'HCO3', 'Hgb', 'Hct']
    
    ALL_LAB_COLS = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 
                   'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 
                   'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 
                   'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets',
                   'EtCO2', 'Bilirubin_direct', 'TroponinI', 'Fibrinogen']

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            config = read_yaml(config_path)
            self.fe_cfg = config.get("feature_engineering", {})
            self.patient_col = config["data_split"].get("patient_column", "Patient_ID")
            self.target_col = config["data_validation"]["target_column"]
            
            # Feature toggles (all enabled by default)
            self.enable_lag = self.fe_cfg.get("enable_lag_features", True)
            self.enable_delta = self.fe_cfg.get("enable_delta_features", True)
            self.enable_rolling = self.fe_cfg.get("enable_rolling_features", True)
            self.enable_clinical = self.fe_cfg.get("enable_clinical_scores", True)
            self.enable_missingness = self.fe_cfg.get("enable_missingness_features", True)
            
            logger.info("AdvancedFeatureEngineer initialized successfully.")
            
        except Exception as e:
            raise CustomException(e, sys)

    def _get_available_cols(self, df: pd.DataFrame, col_list: List[str]) -> List[str]:
        """Return only columns that exist in the dataframe."""
        return [c for c in col_list if c in df.columns]

    # =========================================================================
    # LAG FEATURES (previous values at t-1, t-3, t-6 hours)
    # =========================================================================
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for vital signs.
        These capture what the values were 1, 3, 6 hours ago.
        """
        try:
            logger.info("Creating lag features...")
            
            vital_cols = self._get_available_cols(df, self.VITAL_COLS)
            lag_periods = [1, 3, 6]
            
            for col in vital_cols:
                for lag in lag_periods:
                    df[f'{col}_lag_{lag}h'] = df.groupby(self.patient_col)[col].shift(lag)
            
            n_features = len(vital_cols) * len(lag_periods)
            logger.info(f"Created {n_features} lag features")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # DELTA FEATURES (rate of change)
    # =========================================================================
    def _add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create delta features showing change from previous timepoints.
        Captures deterioration or improvement trends.
        """
        try:
            logger.info("Creating delta (change) features...")
            
            vital_cols = self._get_available_cols(df, self.VITAL_COLS)
            delta_periods = [1, 3, 6]
            
            for col in vital_cols:
                for period in delta_periods:
                    lag_col = f'{col}_lag_{period}h'
                    if lag_col in df.columns:
                        df[f'{col}_delta_{period}h'] = df[col] - df[lag_col]
                    else:
                        # Create lag if not exists
                        lag_values = df.groupby(self.patient_col)[col].shift(period)
                        df[f'{col}_delta_{period}h'] = df[col] - lag_values
            
            n_features = len(vital_cols) * len(delta_periods)
            logger.info(f"Created {n_features} delta features")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # ROLLING STATISTICS (trends over windows)
    # =========================================================================
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics.
        Captures trends and variability over time.
        """
        try:
            logger.info("Creating rolling statistics features...")
            
            vital_cols = self._get_available_cols(df, self.VITAL_COLS)
            windows = [6, 12, 24]  # hours
            
            feature_count = 0
            for col in vital_cols:
                for window in windows:
                    # Rolling mean
                    df[f'{col}_rolling_mean_{window}h'] = df.groupby(self.patient_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    
                    # Rolling std (variability)
                    df[f'{col}_rolling_std_{window}h'] = df.groupby(self.patient_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=2).std()
                    )
                    
                    # Rolling min/max
                    df[f'{col}_rolling_min_{window}h'] = df.groupby(self.patient_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).min()
                    )
                    df[f'{col}_rolling_max_{window}h'] = df.groupby(self.patient_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).max()
                    )
                    
                    feature_count += 4
            
            logger.info(f"Created {feature_count} rolling features")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # CLINICAL SCORES
    # =========================================================================
    def _add_clinical_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinical severity scores used in sepsis detection.
        """
        try:
            logger.info("Creating clinical score features...")
            feature_count = 0
            
            # 1. Shock Index = HR / SBP (higher = worse)
            if 'HR' in df.columns and 'SBP' in df.columns:
                df['Shock_Index'] = df['HR'] / (df['SBP'] + 1e-6)
                # Shock Index > 1 is concerning
                df['Shock_Index_High'] = (df['Shock_Index'] > 1.0).astype(int)
                feature_count += 2
            
            # 2. Modified Shock Index = HR / MAP
            if 'HR' in df.columns and 'MAP' in df.columns:
                df['Modified_Shock_Index'] = df['HR'] / (df['MAP'] + 1e-6)
                feature_count += 1
            
            # 3. Pulse Pressure = SBP - DBP
            if 'SBP' in df.columns and 'DBP' in df.columns:
                df['Pulse_Pressure'] = df['SBP'] - df['DBP']
                feature_count += 1
            
            # 4. Temperature abnormality (fever or hypothermia)
            if 'Temp' in df.columns:
                df['Temp_Abnormal'] = ((df['Temp'] > 38.0) | (df['Temp'] < 36.0)).astype(int)
                df['Fever'] = (df['Temp'] > 38.0).astype(int)
                df['Hypothermia'] = (df['Temp'] < 36.0).astype(int)
                feature_count += 3
            
            # 5. Respiratory distress indicators
            if 'Resp' in df.columns:
                df['Tachypnea'] = (df['Resp'] > 22).astype(int)  # qSOFA criterion
                feature_count += 1
            
            if 'O2Sat' in df.columns:
                df['Hypoxemia'] = (df['O2Sat'] < 90).astype(int)
                feature_count += 1
            
            # 6. Blood pressure indicators
            if 'SBP' in df.columns:
                df['Hypotension'] = (df['SBP'] <= 100).astype(int)  # qSOFA criterion
                df['Severe_Hypotension'] = (df['SBP'] <= 90).astype(int)
                feature_count += 2
            
            if 'MAP' in df.columns:
                df['MAP_Low'] = (df['MAP'] < 65).astype(int)  # Sepsis criterion
                feature_count += 1
            
            # 7. Heart rate abnormalities
            if 'HR' in df.columns:
                df['Tachycardia'] = (df['HR'] > 90).astype(int)
                df['Severe_Tachycardia'] = (df['HR'] > 120).astype(int)
                df['Bradycardia'] = (df['HR'] < 60).astype(int)
                feature_count += 3
            
            # 8. WBC abnormalities (infection markers)
            if 'WBC' in df.columns:
                df['WBC_Abnormal'] = ((df['WBC'] > 12) | (df['WBC'] < 4)).astype(int)
                df['Leukocytosis'] = (df['WBC'] > 12).astype(int)
                df['Leukopenia'] = (df['WBC'] < 4).astype(int)
                feature_count += 3
            
            # 9. Lactate elevation (tissue hypoxia)
            if 'Lactate' in df.columns:
                df['Lactate_Elevated'] = (df['Lactate'] > 2.0).astype(int)
                df['Lactate_High'] = (df['Lactate'] > 4.0).astype(int)  # Severe
                feature_count += 2
            
            # 10. Kidney function (Creatinine)
            if 'Creatinine' in df.columns:
                df['Creatinine_Elevated'] = (df['Creatinine'] > 1.2).astype(int)
                feature_count += 1
            
            # 11. qSOFA Score (quick SOFA) - key sepsis screening tool
            # Criteria: Resp >= 22, SBP <= 100, Altered mental status (not available)
            qsofa_parts = []
            if 'Tachypnea' in df.columns:
                qsofa_parts.append('Tachypnea')
            if 'Hypotension' in df.columns:
                qsofa_parts.append('Hypotension')
            
            if len(qsofa_parts) >= 2:
                df['qSOFA_partial'] = df[qsofa_parts].sum(axis=1)
                feature_count += 1
            
            logger.info(f"Created {feature_count} clinical score features")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # MISSINGNESS FEATURES (clinical concern proxy)
    # =========================================================================
    def _add_missingness_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on data availability.
        More labs ordered often indicates clinical concern.
        """
        try:
            logger.info("Creating missingness indicator features...")
            feature_count = 0
            
            # Count of available vital signs
            vital_cols = self._get_available_cols(df, self.VITAL_COLS)
            if vital_cols:
                df['vital_count'] = df[vital_cols].notna().sum(axis=1)
                df['vital_missing_count'] = df[vital_cols].isna().sum(axis=1)
                feature_count += 2
            
            # Count of available lab values (key labs)
            key_lab_cols = self._get_available_cols(df, self.KEY_LAB_COLS)
            if key_lab_cols:
                df['key_lab_count'] = df[key_lab_cols].notna().sum(axis=1)
                df['key_lab_missing_count'] = df[key_lab_cols].isna().sum(axis=1)
                feature_count += 2
            
            # Count of all available lab values
            all_lab_cols = self._get_available_cols(df, self.ALL_LAB_COLS)
            if all_lab_cols:
                df['all_lab_count'] = df[all_lab_cols].notna().sum(axis=1)
                # Labs ordered is a strong signal - doctors order more when concerned
                df['labs_ordered_flag'] = (df['all_lab_count'] > 3).astype(int)
                feature_count += 2
            
            # Critical lab availability flags
            critical_labs = ['Lactate', 'WBC', 'Creatinine', 'Platelets']
            for lab in critical_labs:
                if lab in df.columns:
                    df[f'{lab}_available'] = df[lab].notna().astype(int)
                    feature_count += 1
            
            logger.info(f"Created {feature_count} missingness features")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # TIME-BASED FEATURES
    # =========================================================================
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        """
        try:
            logger.info("Creating time-based features...")
            feature_count = 0
            
            # Hours since admission (ICULOS is this directly)
            if 'ICULOS' in df.columns:
                df['hours_since_admission'] = df['ICULOS']
                
                # Time bands
                df['early_admission'] = (df['ICULOS'] <= 6).astype(int)  # First 6 hours
                df['first_day'] = (df['ICULOS'] <= 24).astype(int)
                df['first_two_days'] = (df['ICULOS'] <= 48).astype(int)
                
                # Log transform of time (diminishing importance)
                df['log_hours'] = np.log1p(df['ICULOS'])
                
                feature_count += 5
            
            # Hour of day (if Hour column exists)
            if 'Hour' in df.columns:
                df['hour_of_day'] = df['Hour'] % 24
                df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
                feature_count += 2
            
            # Hospital admission time
            if 'HospAdmTime' in df.columns:
                df['hosp_adm_hours'] = df['HospAdmTime'].abs()  # Negative values = before ICU
                df['transferred_from_floor'] = (df['HospAdmTime'] < 0).astype(int)
                feature_count += 2
            
            logger.info(f"Created {feature_count} time-based features")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    # =========================================================================
    # MAIN APPLY FUNCTION
    # =========================================================================
    def apply_features(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Apply all configured feature engineering steps.
        
        Args:
            df: Input dataframe with patient time-series data
            verbose: Whether to log progress
            
        Returns:
            DataFrame with all new features added
        """
        try:
            logger.info("=" * 60)
            logger.info("APPLYING ADVANCED FEATURE ENGINEERING")
            logger.info("=" * 60)
            
            initial_cols = len(df.columns)
            df_fe = df.copy()
            
            # Ensure sorted by patient and time
            if self.patient_col in df_fe.columns:
                time_col = 'ICULOS' if 'ICULOS' in df_fe.columns else 'Hour'
                if time_col in df_fe.columns:
                    df_fe = df_fe.sort_values([self.patient_col, time_col])
            
            # Apply feature engineering steps
            if self.enable_lag:
                df_fe = self._add_lag_features(df_fe)
            
            if self.enable_delta:
                df_fe = self._add_delta_features(df_fe)
            
            if self.enable_rolling:
                df_fe = self._add_rolling_features(df_fe)
            
            if self.enable_clinical:
                df_fe = self._add_clinical_scores(df_fe)
            
            if self.enable_missingness:
                df_fe = self._add_missingness_features(df_fe)
            
            # Always add time features
            df_fe = self._add_time_features(df_fe)
            
            # Summary
            new_cols = len(df_fe.columns) - initial_cols
            logger.info("=" * 60)
            logger.info(f"FEATURE ENGINEERING COMPLETE")
            logger.info(f"Original columns: {initial_cols}")
            logger.info(f"New columns: {new_cols}")
            logger.info(f"Final columns: {len(df_fe.columns)}")
            logger.info("=" * 60)
            
            return df_fe
            
        except Exception as e:
            raise CustomException(e, sys)

    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Return dictionary of feature names by category.
        Useful for feature selection and analysis.
        """
        return {
            'vital_signs': self.VITAL_COLS,
            'key_labs': self.KEY_LAB_COLS,
            'all_labs': self.ALL_LAB_COLS,
        }


# Backward compatibility with old interface
class FeatureEngineer(AdvancedFeatureEngineer):
    """Alias for backward compatibility."""
    pass
