from fastapi import FastAPI, Body
import uvicorn
from pydantic import BaseModel
import pickle

# Créer des tags pour une meilleure organisation de l'API
tags = [
    {"name": "Hello", "description": "Bonjour le monde"},
    {"name": "Prédiction V1", "description": "Prédictions en utilisant le Modèle 1"},
    {"name": "Prédiction V2", "description": "Prédictions en utilisant le Modèle 2"},
]

# Créer l'application FastAPI
app = FastAPI(
    title="API de prédiction",
    description="Prédictions",
    version="1.0.0",
    openapi_tags=tags,
)

# Charger les modèles au démarrage pour une efficacité accrue (considérer le caching pour les gros modèles)
try:
    with open("model_1.pkl", "rb") as f:
        model_1 = pickle.load(f)
    with open("model_2.pkl", "rb") as f:
        model_2 = pickle.load(f)
except FileNotFoundError as e:
    print(f"Erreur lors du chargement des modèles : {e}")
    raise

# Définir les modèles de données pour la validation et la clarté
class Crédit(BaseModel):
    Genre: int
    Âge: int
    Niveau_Activité_Physique: int
    Fréquence_Cardiaque: int
    Pas_Quotidiens: int
    Pression_Artérielle_Haute: int
    Pression_Artérielle_Basse: int

class Santé(BaseModel):
    Niveau_Activité_Physique: int
    Fréquence_Cardiaque: int
    Pas_Quotidiens: int

# Gestion des erreurs avec des réponses informatives
@app.post("/prédire", tags=["Prédiction V1"])
async def prédire_modèle1(crédit: Crédit = Body(...)):
    """
    Prédit en utilisant le Modèle 1 basé sur les données de crédit.
    
    Args:
        crédit (Crédit): Données de crédit pour la prédiction.
    
    Returns:
        dict: Un dictionnaire contenant la prédiction.
    """
    try:
        données = [[crédit.Genre, crédit.Âge, crédit.Niveau_Activité_Physique, crédit.Fréquence_Cardiaque, crédit.Pas_Quotidiens, crédit.Pression_Artérielle_Haute, crédit.Pression_Artérielle_Basse]]
        prédiction = model_1.predict(données)[0]  # En supposant qu'il retourne une seule prédiction
        return {"prédiction": int(prédiction)}  # Convertir la prédiction en type Python natif
    except Exception as e:
        return {"erreur": str(e)}

@app.post("/prédire2", tags=["Prédiction V2"])
async def prédire_modèle2(santé: Santé = Body(...)):
    """
    Prédit en utilisant le Modèle 2 basé sur les données de santé.
    
    Args:
        santé (Santé): Données de santé pour la prédiction.
    
    Returns:
        dict: Un dictionnaire contenant la prédiction.
    """
    try:
        données = [[santé.Niveau_Activité_Physique, santé.Fréquence_Cardiaque, santé.Pas_Quotidiens]]
        prédiction = model_2.predict(données)[0]  # En supposant qu'il retourne une seule prédiction
        return {"prédiction": int(prédiction)}  # Convertir la prédiction en type Python natif
    except Exception as e:
        return {"erreur": str(e)}

# Démarrer le serveur Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
