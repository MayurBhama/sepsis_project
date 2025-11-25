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


    def connect_mongo(self):
        """Connect to MongoDB using URI from config."""
        try:
            logger.info("Connecting to MongoDB...")

            client = MongoClient(self.config["mongodb"]["uri"])
            db = client[self.config["mongodb"]["database"]]
            collection = db[self.config["mongodb"]["raw_collection"]]

            logger.info("MongoDB connection successful.")
            return collection

        except Exception as e:
            logger.error("MongoDB connection failed.")
            raise CustomException(e, sys)


    def ingest_csv(self):
        """Reads CSV and inserts data to MongoDB."""
        try:
            csv_path = self.config["data_source"]["csv_path"]
            logger.info(f"Reading CSV file from: {csv_path}")

            df = pd.read_csv(csv_path)
            logger.info(f"CSV Loaded. Shape: {df.shape}")

            collection = self.connect_mongo()

            logger.info("Converting DataFrame to JSON records...")
            records = df.to_dict(orient="records")

            logger.info("Inserting records into MongoDB...")
            collection.insert_many(records)

            logger.info(f"Data ingestion completed successfully. Inserted {len(records)} rows.")

            return df  # Returning df for next pipeline steps

        except Exception as e:
            logger.error("Error occurred during CSV ingestion.")
            raise CustomException(e, sys)



if __name__ == "__main__":

    try:
        ingestion = DataIngestion()
        ingestion.ingest_csv()

    except Exception as e:
        print(e)