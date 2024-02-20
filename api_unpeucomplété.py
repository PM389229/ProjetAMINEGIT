from fastapi import FastAPI, Body
import uvicorn
from pydantic import BaseModel
import pickle
import numpy as np

# Create tags for better API organization
tags = [
    {"name": "Hello", "description": "Hello World"},
    {"name": "Predict V1", "description": "Predictions using Model 1"},
    {"name": "Predict V2", "description": "Predictions using Model 2"},
]

# Create the FastAPI application
app = FastAPI(
    title="API de prediction",
    description="Predictions",
    version="1.0.0",
    openapi_tags=tags,
)

# Load models at startup for efficiency (consider caching for large models)
try:
    with open("model_1.pkl", "rb") as f:
        model_1 = pickle.load(f)
    with open("model_2.pkl", "rb") as f:
        model_2 = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    raise

# Define data models for validation and clarity
class Credit(BaseModel):
    Gender: int
    Age: int
    Physical_Activity_Level: int
    Heart_Rate: int
    Daily_Steps: int
    BloodPressure_high: int
    BloodPressure_low: int

class Health(BaseModel):
    Physical_Activity_Level: int
    Heart_Rate: int
    Daily_Steps: int

# Error handling with informative responses
@app.post("/predict", tags=["Predict V1"])
async def predict_model1(credit: Credit = Body(...)):
    try:
        data = [[credit.Gender, credit.Age, credit.Physical_Activity_Level, credit.Heart_Rate, credit.Daily_Steps, credit.BloodPressure_high, credit.BloodPressure_low]]
        prediction = model_1.predict(data)[0]  # Assuming it returns single prediction
        return {"prediction": int(prediction)}  # Convert prediction to native Python type
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict2", tags=["Predict V2"])
async def predict_model2(health: Health = Body(...)):
    try:
        data = [[health.Physical_Activity_Level, health.Heart_Rate, health.Daily_Steps]]
        prediction = model_2.predict(data)[0]  # Assuming it returns single prediction
        return {"prediction": int(prediction)}  # Convert prediction to native Python type
    except Exception as e:
        return {"error": str(e)}

# Start the Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
