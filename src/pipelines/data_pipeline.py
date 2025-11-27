import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException

from src.data_ingestion.ingest_to_mongo import DataIngestion
from src.data_validation.data_validation import DataValidator
from src.preprocessing.preprocessor import DataPreprocessor
from src.feature_engineering.feature_engineer import FeatureEngineer


def run_data_pipeline():
    try:
        logger.info("========== DATA PIPELINE STARTED ==========")

        # STEP 0: INIT MODULES
        ingestor = DataIngestion()
        validator = DataValidator()
        preprocessor = DataPreprocessor()
        fe = FeatureEngineer()

        # STEP 1: FETCH RAW DATA FROM MONGODB
        logger.info("Fetching raw data from MongoDB...")
        df_raw = ingestor.fetch_raw()      # FIXED
        logger.info(f"Raw data loaded. Shape: {df_raw.shape}")

        # STEP 2: VALIDATION
        logger.info("Running Data Validation...")
        df_valid = validator.run_validation(df_raw)
        logger.info(f"Validation completed. Shape: {df_valid.shape}")

        # STEP 3: PREPROCESSING
        logger.info("Running Preprocessing...")
        df_clean = preprocessor.run_preprocessing(df_valid)
        logger.info(f"Preprocessing completed. Shape: {df_clean.shape}")

        # STEP 4: FEATURE ENGINEERING
        logger.info("Applying Feature Engineering...")
        df_final = fe.apply_features(df_clean)
        logger.info(f"Feature engineering completed. Shape: {df_final.shape}")

        # STEP 5: SAVE CLEAN DATA BACK TO MONGODB
        logger.info("Saving cleaned data back to MongoDB...")
        ingestor.save_clean(df_final)      # FIXED
        logger.info("Clean data saved into clean_sepsis_data collection successfully.")

        # STEP 6 (OPTIONAL): SAVE AS CSV
        preprocessor.save_preprocessed(df_final)

        logger.info("========== DATA PIPELINE COMPLETED SUCCESSFULLY ==========")

        return df_final

    except Exception as e:
        raise CustomException(e, sys)


# Run using:
# python -m src.pipelines.data_pipeline
if __name__ == "__main__":
    run_data_pipeline()
