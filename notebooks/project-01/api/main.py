import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import logging
import pickle
from io import BytesIO

import pandas as pd
from config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.s3_utils import S3Utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
s3_utils = S3Utils(config)
app = FastAPI()

# Load model and preprocessor
try:
    logger.info("Loading model and preprocessor from S3...")
    model_path = Path(config.PATHS["model_dir"]) / "best_model.joblib"
    preprocessor_path = Path(config.PATHS["model_dir"]) / "feature_preprocessor.joblib"

    s3_utils.download_file("best_model.joblib", str(model_path))
    s3_utils.download_file("feature_preprocessor.joblib", str(preprocessor_path))

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(preprocessor_path, "rb") as f:
        feature_preprocessor = pickle.load(f)

    logger.info("Model and preprocessor loaded successfully")
except Exception as e:
    logger.exception(f"An error occurred while loading model or preprocessor: {str(e)}")
    raise


class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_blood_pressure: int
    serum_cholestoral: int
    fasting_blood_sugar: int
    resting_electrocardiographic_results: int
    max_heart_rate: int
    exercise_induced_angina: int
    oldpeak: float
    st_segment: int
    major_vessels: int
    thal: int

    class Config:
        schema_extra = {
            "example": {
                "age": 70,
                "sex": 1,
                "chest_pain_type": 4,
                "resting_blood_pressure": 130,
                "serum_cholestoral": 322,
                "fasting_blood_sugar": 0,
                "resting_electrocardiographic_results": 2,
                "max_heart_rate": 109,
                "exercise_induced_angina": 0,
                "oldpeak": 2.4,
                "st_segment": 2,
                "major_vessels": 3,
                "thal": 3,
            }
        }


@app.post("/predict")
async def predict_heart_disease(input: HeartDiseaseInput):
    try:
        input_df = pd.DataFrame([input.dict()])
        X_processed = feature_preprocessor.transform(input_df)
        prediction = model.predict(X_processed)
        probability = (
            model.predict_proba(X_processed)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        result = {"prediction": int(prediction[0])}
        if probability is not None:
            result["probability"] = float(probability[0])

        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    if model is not None and feature_preprocessor is not None:
        return {"status": "healthy", "message": "Model and preprocessor are loaded"}
    raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
