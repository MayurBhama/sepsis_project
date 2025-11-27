from src.logger import logger
from src.exception import CustomException
from src.model_training.trainer import ModelTrainer
import sys

def run_training_pipeline():
    try:
        logger.info("===== TRAINING PIPELINE STARTED =====")
        trainer = ModelTrainer()
        trainer.train()
        logger.info("===== TRAINING PIPELINE COMPLETED SUCCESSFULLY =====")
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_training_pipeline()