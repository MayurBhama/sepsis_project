import os
import sys
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml


class DataPreprocessor:
    """
    - handle missing values (imputation)
    - remove duplicates
    - cap outliers (1st–99th percentile)
    - encode categoricals
    - feature engineering (Shock_Index, Temp_Abnormal, WBC_Abnormal)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            self.validation_cfg = self.config["data_validation"]
            self.preproc_cfg = self.config["preprocessing"]

            self.target_col = self.validation_cfg["target_column"]
            self.numerical_cols_cfg = self.validation_cfg["numerical_columns"]
            self.categorical_cols_cfg = self.validation_cfg.get("categorical_columns", [])

            self.lower_pct = self.preproc_cfg["outlier"]["lower_percentile"]
            self.upper_pct = self.preproc_cfg["outlier"]["upper_percentile"]
            self.output_path = self.preproc_cfg["output_path"]
            self.fe_cfg = self.preproc_cfg["feature_engineering"]

            logger.info("DataPreprocessor initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Median imputation for numerical, mode for categorical."""
        try:
            logger.info("Handling missing values (imputation)...")

            df_processed = df.copy()

            # numerical columns (intersection of config + actual columns)
            num_cols = [c for c in self.numerical_cols_cfg if c in df_processed.columns]
            # exclude target from imputation
            num_cols = [c for c in num_cols if c != self.target_col]

            cat_cols = [c for c in self.categorical_cols_cfg if c in df_processed.columns]

            # Numerical imputation
            if num_cols:
                num_imputer = SimpleImputer(strategy="median")
                df_processed[num_cols] = num_imputer.fit_transform(df_processed[num_cols])
                logger.info(f"Imputed numerical columns (median): {num_cols}")
            else:
                logger.info("No numerical columns found for imputation.")

            # Categorical imputation
            if cat_cols:
                cat_imputer = SimpleImputer(strategy="most_frequent")
                df_processed[cat_cols] = cat_imputer.fit_transform(df_processed[cat_cols])
                logger.info(f"Imputed categorical columns (most_frequent): {cat_cols}")
            else:
                logger.info("No categorical columns found for imputation.")

            remaining_missing = df_processed.isnull().sum().sum()
            logger.info(f"Total remaining missing values after imputation: {remaining_missing}")
            return df_processed

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        try:
            logger.info("Removing duplicate rows...")
            before = len(df)
            df_no_dup = df.drop_duplicates()
            after = len(df_no_dup)
            logger.info(f"Removed {before - after} duplicate rows. Final rows: {after}")
            return df_no_dup
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers at configured percentiles (1st and 99th by default)."""
        try:
            logger.info(
                f"Capping outliers for numerical columns between "
                f"{self.lower_pct} and {self.upper_pct} percentiles."
            )

            df_capped = df.copy()
            num_cols = [c for c in self.numerical_cols_cfg if c in df_capped.columns and c != self.target_col]

            for col in num_cols:
                lower_cap = df_capped[col].quantile(self.lower_pct / 100)
                upper_cap = df_capped[col].quantile(self.upper_pct / 100)

                outliers = ((df_capped[col] < lower_cap) | (df_capped[col] > upper_cap)).sum()
                df_capped[col] = df_capped[col].clip(lower=lower_cap, upper=upper_cap)

                logger.info(
                    f"{col}: capped {outliers} values to "
                    f"[{lower_cap:.2f}, {upper_cap:.2f}]"
                )

            return df_capped

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label encode binary categoricals; one-hot encode multi-class."""
        try:
            logger.info("Encoding categorical variables (if any)...")
            df_encoded = df.copy()

            # If you ever have real categorical columns, this logic is ready.
            cat_cols = [c for c in self.categorical_cols_cfg if c in df_encoded.columns]

            if not cat_cols:
                logger.info("No categorical columns configured for encoding.")
                return df_encoded

            for col in cat_cols:
                unique_vals = df_encoded[col].nunique()
                if unique_vals == 2:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    logger.info(f"{col}: binary label encoded.")
                else:
                    df_encoded = pd.get_dummies(
                        df_encoded, columns=[col], prefix=col, drop_first=True
                    )
                    logger.info(f"{col}: one-hot encoded with {unique_vals} categories.")

            return df_encoded

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Shock_Index (HR/SBP)
        - Temp_Abnormal (if Temp exists)
        - WBC_Abnormal (if WBC exists)
        """
        try:
            logger.info("Applying feature engineering...")
            fe_done = False
            df_fe = df.copy()

            # Shock Index
            if (
                self.fe_cfg.get("create_shock_index", False)
                and "HR" in df_fe.columns
                and "SBP" in df_fe.columns
            ):
                df_fe["Shock_Index"] = df_fe["HR"] / (df_fe["SBP"] + 1e-6)
                logger.info("Created feature: Shock_Index (HR/SBP)")
                fe_done = True

            # Temp_Abnormal (note: Temp column was dropped in validation, but logic preserved)
            if (
                self.fe_cfg.get("create_temp_abnormal", False)
                and "Temp" in df_fe.columns
            ):
                df_fe["Temp_Abnormal"] = (
                    (df_fe["Temp"] > 38) | (df_fe["Temp"] < 36)
                ).astype(int)
                logger.info("Created feature: Temp_Abnormal")
                fe_done = True
            else:
                logger.info("Temp_Abnormal skipped (Temp column not present).")

            # WBC_Abnormal (WBC was dropped due to high missingness, but logic preserved)
            if (
                self.fe_cfg.get("create_wbc_abnormal", False)
                and "WBC" in df_fe.columns
            ):
                df_fe["WBC_Abnormal"] = (
                    (df_fe["WBC"] > 12) | (df_fe["WBC"] < 4)
                ).astype(int)
                logger.info("Created feature: WBC_Abnormal")
                fe_done = True
            else:
                logger.info("WBC_Abnormal skipped (WBC column not present).")

            if not fe_done:
                logger.info("No feature engineering features created (conditions not met).")

            return df_fe

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def save_preprocessed(self, df: pd.DataFrame) -> None:
        """Save preprocessed dataframe to CSV."""
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            df.to_csv(self.output_path, index=False)
            logger.info(f"Preprocessed data saved to: {self.output_path}")
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def run_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing pipeline (Notebook Section 4 & 5)."""
        try:
            logger.info("Starting Data Preprocessing Pipeline...")

            df_proc = self._handle_missing_values(df)
            df_proc = self._remove_duplicates(df_proc)
            df_proc = self._cap_outliers(df_proc)
            df_proc = self._encode_categoricals(df_proc)
            df_proc = self._feature_engineering(df_proc)

            logger.info(f"Preprocessing completed. Final shape: {df_proc.shape}")
            return df_proc

        except Exception as e:
            raise CustomException(e, sys)
