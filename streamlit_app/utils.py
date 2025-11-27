import pymongo
from datetime import datetime

def get_mongo_client(uri):
    return pymongo.MongoClient(uri)

def log_prediction(db, input_data, result, source="streamlit"):
    record = {
        "timestamp": datetime.utcnow(),
        "inputs": input_data,
        "probability": result["probability"],
        "label": result["predicted_label"],
        "threshold": result["threshold_used"],
        "source": source
    }
    db.insert_one(record)