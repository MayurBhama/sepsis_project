import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml


class FeatureEngineer:
    """
    Creates new medical features:
    - Shock_Index = HR / SBP
    - Temp_Abnormal = (Temp > 38 or < 36)
    - WBC_Abnormal = (WBC > 12 or < 4)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            config = read_yaml(config_path)
            self.fe_cfg = config["preprocessing"]["feature_engineering"]
            logger.info("FeatureEngineer initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured feature engineering steps."""
        try:
            logger.info("Applying Feature Engineering...")
            df_fe = df.copy()

            # 1. Shock Index ---------------------------------------
            if (
                self.fe_cfg.get("create_shock_index", False)
                and "HR" in df_fe.columns
                and "SBP" in df_fe.columns
            ):
                df_fe["Shock_Index"] = df_fe["HR"] / (df_fe["SBP"] + 1e-6)
                logger.info("Created feature: Shock_Index")

            # 2. Temp_Abnormal --------------------------------------
            if (
                self.fe_cfg.get("create_temp_abnormal", False)
                and "Temp" in df_fe.columns
            ):
                df_fe["Temp_Abnormal"] = (
                    (df_fe["Temp"] > 38) | (df_fe["Temp"] < 36)
                ).astype(int)
                logger.info("Created feature: Temp_Abnormal")
            else:
                logger.info("Temp_Abnormal skipped (Temp missing or disabled).")

            # 3. WBC_Abnormal ---------------------------------------
            if (
                self.fe_cfg.get("create_wbc_abnormal", False)
                and "WBC" in df_fe.columns
            ):
                df_fe["WBC_Abnormal"] = (
                    (df_fe["WBC"] > 12) | (df_fe["WBC"] < 4)
                ).astype(int)
                logger.info("Created feature: WBC_Abnormal")
            else:
                logger.info("WBC_Abnormal skipped (WBC missing or disabled).")

            logger.info(f"Feature Engineering complete. Final shape: {df_fe.shape}")
            return df_fe

        except Exception as e:
            raise CustomException(e, sys)
