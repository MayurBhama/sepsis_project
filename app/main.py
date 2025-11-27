# app/main.py

from fastapi import FastAPI
from src.logger import logger

from app.schemas import SepsisRequest, SepsisResponse
from app.predictor import predictor


app = FastAPI(
    title="Sepsis Prediction API",
    description="Predicts probability of sepsis using trained XGBoost model.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """
    Simple health endpoint for monitoring.
    """
    logger.info("Health check called.")
    return {"status": "ok"}


@app.post("/predict", response_model=SepsisResponse)
def predict_sepsis(request: SepsisRequest):
    """
    Take patient vitals/labs and return prediction output:
    - probability
    - predicted_label (0/1)
    - threshold_used
    """
    logger.info("Received prediction request.")

    payload = request.dict()

    result = predictor.predict(payload)

    return SepsisResponse(
        probability=result["probability"],
        predicted_label=result["predicted_label"],
        threshold_used=float(result["threshold_used"]),
    )
