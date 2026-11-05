from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model, ConfigDict

BASE_DIR = Path(__file__).resolve().parent
artifact = joblib.load(BASE_DIR / "breast_cancer_pipeline.joblib")
pipeline = artifact["pipeline"]
FEATURE_NAMES = artifact["feature_names"]
TARGET_NAMES = {int(k): v for k, v in artifact["target_names"].items()}
# TARGET_NAMES = {
#     0: "Benign",
#     1: "Malignant"
# }

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Educational demonstration only; not for medical diagnosis.",
    version="1.0.0"
)

class PredictionInput(BaseModel):
    features: dict[str, float]

@app.get("/")
def home():
    return {
        "message": "ML Prediction API is running",
        "docs": "/docs",
        "prediction_endpoint": "/predict"
    }

@app.post("/predict")
def predict(payload: PredictionInput):
    received = payload.features
    missing = [name for name in FEATURE_NAMES if name not in received]
    extra = [name for name in received if name not in FEATURE_NAMES]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Input features do not match the model requirements",
                "missing_features": missing,
                "unexpected_features": extra
            }
        )

    try:
        input_df = pd.DataFrame(
            [[received[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES
        )
        prediction = int(pipeline.predict(input_df)[0])
        probabilities = pipeline.predict_proba(input_df)[0]

        return {
            "prediction": TARGET_NAMES[prediction],
            "class_id": prediction,
            "probabilities": {
                TARGET_NAMES[int(class_id)]: round(float(prob), 4)
                for class_id, prob in zip(pipeline.classes_, probabilities)
            } 
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
