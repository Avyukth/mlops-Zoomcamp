import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model and preprocessor
model = joblib.load("models/best_model.joblib")
feature_preprocessor = joblib.load("models/feature_preprocessor.joblib")

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
                "thal": 3
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
        logger.info(f"Probability: {prediction}")
        # Get probability if the model supports it
        if hasattr(model, 'predict_proba'):
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
