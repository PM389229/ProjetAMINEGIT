# Import des librairies uvicorn, pickle, FastAPI, File, UploadFile, BaseModel
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np 
from pydantic import BaseModel
import pickle
import pandas as pd

import mlflow
import os
import boto3



# Création des tags
tags = [
       {
              "name": "Hello",
              "description": "Hello World",
       },
       {
              "name": "Predict",
              "description": "Predict",
       },
]

# Création de l'application
app = FastAPI(
       title="API de prediction",
       description= "Predictions",
       version= "1.0.0",
       openapi_tags= tags
)

# Point de terminaison avec paramètre
@app.get("/hello", tags=["Hello name V1"])
def hello(name: str='World'):
        return {"message": f"Hello {name}"}



# Création du modèle de données pour le modéle 1 ('Gender', 'Age', 'Physical Activity Level', 'Heart Rate', 'Daily Steps', 'BloodPressure_high', 'BloodPressure_low', 'Sleep Disorder'])
class Credit(BaseModel):
        Gender : int
        Age : int
        ...

# Point de terminaison : Prédiction 1
@app.post("/predict", tags=["Predict V1"])
def predict(credit: Credit) :
       ...

# Création du modèle de données pour le modéle 2 ('Physical Activity Level', 'Heart Rate', 'Daily Steps', 'Sleep Disorder')


# Point de terminaison : Prédiction 2

# Démarage de l'application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)