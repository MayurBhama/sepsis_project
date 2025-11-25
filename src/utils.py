import yaml
import sys 
from src.logger import logger 
from src.exception import CustomException

def read_yaml(file_path: str):
    try:
        logger.info(f"Reading YAML file from: {file_path}")

        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        logger.info(f"YAML file loaded successfully")
        return data
    
    except FileNotFoundError:
        logger.error(f"YAML file not found at: {file_path}")
        raise CustomException(f"YAML file missing: {file_path}", sys)
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {file_path}")
        raise CustomException(f"Invalid YAML format in: {file_path}", sys)
    
    except Exception as e:
        logger.error("Unexpected error while reading YAML file")
        raise CustomException(e, sys)