import sys
import pandas as pd
from pymongo import MongoClient

from src.logger import logger
from src.exception import CustomException
from src.utils import read_yaml


class DataIngestion:

    def __init__(self, config_path="config/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            logger.info("DataIngestion initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    # CONNECT TO MONGO
    # -------------------------------------------------
    def _get_collection(self, collection_name: str):
        """Internal helper to return a Mongo collection."""
        try:
            client = MongoClient(self.config["mongodb"]["uri"])
            db = client[self.config["mongodb"]["database"]]
            collection = db[collection_name]
            return collection
        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    # INGEST CSV → INSERT INTO raw_collection
    # -------------------------------------------------
    def ingest_csv(self):
        """Reads CSV and inserts raw data to MongoDB."""
        try:
            csv_path = self.config["data_source"]["csv_path"]
            logger.info(f"Reading CSV file from: {csv_path}")

            df = pd.read_csv(csv_path)
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")

            raw_collection = self._get_collection(self.config["mongodb"]["raw_collection"])

            logger.info("Converting DataFrame to JSON...")
            records = df.to_dict(orient="records")

            logger.info("Inserting raw data into MongoDB...")
            raw_collection.insert_many(records)

            logger.info(f"Inserted {len(records)} rows into raw_collection.")
            return df

        except Exception as e:
            logger.error("Error occurred during CSV ingestion.")
            raise CustomException(e, sys)

    # -------------------------------------------------
    # FETCH RAW DATA FROM MongoDB
    # -------------------------------------------------
    def fetch_raw(self) -> pd.DataFrame:
        """Fetches raw data stored earlier in MongoDB."""
        try:
            logger.info("Fetching raw data from MongoDB...")

            raw_collection = self._get_collection(self.config["mongodb"]["raw_collection"])

            records = list(raw_collection.find({}, {"_id": 0}))
            df = pd.DataFrame(records)

            logger.info(f"Fetched raw data. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error("Failed to fetch raw data.")
            raise CustomException(e, sys)

    # -------------------------------------------------
    # SAVE CLEANED DATA TO clean_collection
    # -------------------------------------------------
    def save_clean(self, df: pd.DataFrame):
        """Save preprocessed/cleaned data into clean_collection."""
        try:
            logger.info("Saving cleaned data to MongoDB clean_collection...")

            clean_collection = self._get_collection(self.config["mongodb"]["clean_collection"])

            clean_collection.delete_many({})   # Optional: clear old clean data
            clean_collection.insert_many(df.to_dict(orient="records"))

            logger.info(f"Clean data saved successfully. Rows: {df.shape[0]}")

        except Exception as e:
            logger.error("Failed to save clean data.")
            raise CustomException(e, sys)



# -------------------------------------------------
# RUN THIS FILE DIRECTLY → Only for testing ingestion
# -------------------------------------------------
if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        ingestion.ingest_csv()   # loads CSV → mongo
    except Exception as e:
        print(e)