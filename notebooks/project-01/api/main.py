import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import logging
from io import BytesIO

from config import Config, S3Utils
import boto3
import joblib
import pandas as pd
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = Config()
print(config)
s3_utils = S3Utils(config)
app = FastAPI()


# Load configuration


# Function to load model or preprocessor from S3
def load_from_s3(bucket, key):
    try:
        response = s3_utils.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        return joblib.load(BytesIO(body))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.error(f"The file {key} does not exist in the bucket {bucket}")
            raise FileNotFoundError(f"The file {key} does not exist in the S3 bucket")
        else:
            logger.error(f"An error occurred while fetching {key} from S3: {str(e)}")
            raise


# Load model and preprocessor from S3
try:
    logger.info("Loading model from S3...")
    model = load_from_s3(config["s3"]["bucket_name"], "best_model.joblib")
    logger.info("Model loaded successfully")

    logger.info("Loading feature preprocessor from S3...")
    feature_preprocessor = load_from_s3(
        config["s3"]["bucket_name"], "feature_preprocessor.joblib"
    )
    logger.info("Feature preprocessor loaded successfully")
except FileNotFoundError as e:
    logger.error(f"File not found in S3: {str(e)}")
    raise
except Exception as e:
    logger.error(f"An unexpected error occurred: {str(e)}")
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
        # Convert input to DataFrame
        input_dict = input.dict()
        input_df = pd.DataFrame([input_dict])

        # Preprocess the input
        X_processed = feature_preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(X_processed)
        logger.info(f"Prediction: {prediction}")
        # Get probability if the model supports it
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_processed)[:, 1]
            logger.info(f"Probability: {probability[0]}")
        else:
            probability = None

        result = {"prediction": int(prediction[0])}
        if probability is not None:
            result["probability"] = float(probability[0])

        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    try:
        # Check if model and preprocessor are loaded
        if model is not None and feature_preprocessor is not None:
            return {"status": "healthy", "message": "Model and preprocessor are loaded"}
        else:
            raise HTTPException(
                status_code=503, detail="Model or preprocessor not loaded"
            )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
