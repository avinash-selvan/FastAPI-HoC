from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel
import numpy as np

# 1️⃣ Load the trained XGBoost model
model = joblib.load("xgboost_model.pkl")  # Ensure model is saved!

# 2️⃣ Define FastAPI app
app = FastAPI()

# 3️⃣ Define Request Schema (Input Features)
class PredictionRequest(BaseModel):
    reproduction_rate: float
    new_tests_smoothed: float
    positive_rate: float
    new_vaccinations_smoothed: float
    stringency_index: float
    cases_lag_7: float
    cases_ma_7: float

# 4️⃣ Define API Endpoint for Prediction

@app.post("/predict")
def predict(data: PredictionRequest):
    # Convert input to DataFrame
    input_data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return result
    return {"predicted_cases": float(prediction[0])}

# 5️⃣ Run the API server (Only for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

