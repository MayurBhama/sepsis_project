import yaml
import sys 
import os
import pickle
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
    
#------------------------------------------------------------------ 

def save_object(file_path: str, obj) -> None:
    """
    Save Python objects using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

        logger.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    
#------------------------------------------------------------------ 

def load_object(file_path: str):
    """
    Load Python objects using pickle.
    """
    try:
        with open(file_path, "rb") as f:
            loaded_object = pickle.load(f)

        logger.info(f"Object loaded successfully from: {file_path}")
        return loaded_object

    except Exception as e:
        raise CustomException(e, sys)