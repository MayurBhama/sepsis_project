import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import read_yaml
from src.logger import logger
from src.exception import CustomException
import pymongo


def run_split_pipeline(config_path="config/config.yaml"):
    try:
        cfg = read_yaml(config_path)

        mongo_cfg = cfg["mongodb"]
        split_cfg = cfg["data_split"]

        # Load from MongoDB
        client = pymongo.MongoClient(mongo_cfg["uri"])
        db = client[mongo_cfg["database"]]
        df = pd.DataFrame(list(db[mongo_cfg["clean_collection"]].find({}, {"_id": 0})))

        logger.info(f"Loaded clean data for splitting: {df.shape}")

        # Split
        train_df, test_df = train_test_split(
            df,
            test_size=split_cfg["test_size"],
            random_state=split_cfg["random_state"],
            stratify=df[cfg["data_validation"]["target_column"]]
        )

        # Create output directory
        os.makedirs("data/processed", exist_ok=True)

        # Save files
        train_df.to_csv(split_cfg["train_path"], index=False)
        test_df.to_csv(split_cfg["test_path"], index=False)

        logger.info("Train saved: " + split_cfg["train_path"] + " -> " + str(train_df.shape))
        logger.info("Test saved: " + split_cfg["test_path"] + " -> " + str(test_df.shape))
        logger.info("===== DATA SPLIT COMPLETED SUCCESSFULLY =====")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_split_pipeline()