from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
# import joblib  # désactivé pour le déploiement léger
# import uvicorn  # pas nécessaire sur Azure, c'est la plateforme qui lance Uvicorn

# --- Initialisation de l’application ---
app = FastAPI(
    title="Projet UA3 - API de Prédiction - Analyse des factures d’électricité",
    description="API FastAPI (version déploiement  sur Azure).",
    version="1.0"
)

# --- Modèle : version légère pour déploiement sur Azure ---
class DummyModel:
    def predict(self, X):
        # renvoie une valeur fixe, suffisant pour démontrer l'API
        return [123.45] * len(X)

# Si plus tard tu veux utiliser le vrai modèle :
# model = joblib.load("app/model.pkl")

model = DummyModel()

# --- Schéma des données d’entrée ---
class InputData(BaseModel):
    features: list[float]

# --- Routes ---
@app.get("/health")
def health_check():
    """Vérifie si l’API fonctionne"""
    return {"status": "API opérationnelle sur Azure"}

@app.post("/predict")
def predict(data: InputData):
    try:
        X = np.array(data.features).reshape(1, -1)
        prediction = model.predict(X)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}