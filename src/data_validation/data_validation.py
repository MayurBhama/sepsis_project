import pandas as pd 
import sys 
from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml

class DataValidator:

    def __init__(self, config_path= "config/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            self.schema = self.config["data_validation"]
            logger.info("DataValidator initialized with schema")
        except Exception as e:
            raise CustomException(e, sys)
        
    # ---------------------------------------------
    # 1. Check if all required columns exist
    # ---------------------------------------------
    def validate_schema(self, df:pd.DataFrame):
        logger.info("Validating schema....")

        required_cols = (
            self.schema["numerical_columns"]
            + self.schema["categorical_columns"]
            + [self.schema["target_column"]]
        )

        required_cols = set(required_cols)
        df_cols = set(df.columns)

        missing = required_cols - df_cols
        extra = df_cols - required_cols

        if missing:
            raise CustomException(f"Missing columns: {missing}", sys)

        if extra:
            logger.warning(f"Extra columns detected: {extra}")

        logger.info("Schema validation passed.")

    # ---------------------------------------------
    # 2. Detect 100% null columns
    # ---------------------------------------------
    def validate_null_columns(self, df: pd.DataFrame):
        logger.info("Checking for 100% null columns...")

        null_100 = df.columns[df.isnull().sum() == len(df)]

        if len(null_100) > 0:
            logger.warning(f"Columns with 100% null values: {list(null_100)}")

        return list(null_100)

    # ---------------------------------------------
    # 3. Drop columns above missing threshold
    # ---------------------------------------------
    def validate_missing_percentage(self, df: pd.DataFrame):

        threshold = self.schema["max_missing_threshold"]
        logger.info(f"Applying missing-value threshold: {threshold}%")

        percent_missing = df.isnull().mean() * 100
        cols_to_drop = percent_missing[percent_missing > threshold].index.tolist()

        if cols_to_drop:
            logger.warning(f"Dropping columns > {threshold}% missing: {cols_to_drop}")

        remaining_df = df.drop(columns=cols_to_drop)
        return remaining_df, cols_to_drop

    # ---------------------------------------------
    # 4. Validate datatypes
    # ---------------------------------------------
    def validate_dtypes(self, df: pd.DataFrame):
        logger.info("Validating datatypes...")

        # numerical
        for col in self.schema["numerical_columns"]:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise CustomException(f"Column {col} must be numeric.", sys)

        # categorical
        for col in self.schema["categorical_columns"]:
            if col in df.columns:
                if not pd.api.types.is_object_dtype(df[col]):
                    raise CustomException(f"Column {col} must be categorical.", sys)

        logger.info("Datatype validation passed.")

    # ---------------------------------------------
    # COMBINED validation function
    # ---------------------------------------------
    def run_validation(self, df: pd.DataFrame):
        try:
            logger.info("Starting Data Validation Pipeline...")

            self.validate_schema(df)

            null_100_cols = self.validate_null_columns(df)

            df_clean, high_missing_cols = self.validate_missing_percentage(df)

            self.validate_dtypes(df_clean)

            logger.info("All validation checks passed successfully.")

            return df_clean, null_100_cols, high_missing_cols

        except Exception as e:
            raise CustomException(e, sys)