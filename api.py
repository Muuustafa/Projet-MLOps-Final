from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import yaml
import logging
import time
from datetime import datetime
from typing import Dict, Any
import os
from contextlib import asynccontextmanager

# Setup logging structuré pour audit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# Variables globales
model_package = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Gestionnaire de cycle de vie
    global model_package
    
    try:
        if not os.path.exists('models/model.pkl'):
            logging.warning("Modèle non trouvé, entraînement automatique...")
            from train import main as train_main
            train_main()
        
        model_package = joblib.load('models/model.pkl')
        logging.info(f"Modèle chargé: {model_package['model_name']}")
        
        # Log audit démarrage
        audit_log = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'api_startup',
            'model_name': model_package['model_name'],
            'model_performance_r2': model_package['performance']['r2_test']
        }
        logging.info(f"AUDIT: {audit_log}")
        
    except Exception as e:
        logging.error(f"Erreur chargement modèle: {e}")
        logging.info("API démarrée sans modèle")
        model_package = None
    
    yield
    
    # Cleanup
    logging.info("Arrêt API")

app = FastAPI(
    title="House Price Prediction API", 
    version="1.0",
    lifespan=lifespan
)

class HouseData(BaseModel):
    # Données maison pour prédiction
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: bool
    view: int
    condition: int
    sqft_above: int
    sqft_basement: int
    city: str = "Seattle"
    statezip: str = "WA 98101"
    country: str = "USA"

class PredictionResponse(BaseModel):
    # Réponse de prédiction
    predicted_price: float
    model_name: str
    timestamp: str
    processing_time_ms: float

def prepare_features(data: HouseData) -> np.ndarray:
    # On prépare les features pour prédiction
    # Créer DataFrame
    df = pd.DataFrame([{
        'bedrooms': data.bedrooms,
        'bathrooms': data.bathrooms,
        'sqft_living': data.sqft_living,
        'sqft_lot': data.sqft_lot,
        'floors': data.floors,
        'waterfront': 1 if data.waterfront else 0,
        'view': data.view,
        'condition': data.condition,
        'sqft_above': data.sqft_above,
        'sqft_basement': data.sqft_basement,
        'city': data.city,
        'statezip': data.statezip,
        'country': data.country
    }])
    
    # Feature engineering
    df['house_age'] = 2025 - 1990
    df['is_renovated'] = 0
    df['total_sqft'] = df['sqft_living'] + df['sqft_basement']
    
    # Encoder variables catégorielles
    encoders = model_package['encoders']
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform([df[col].iloc[0]])[0]
            except:
                df[col] = 0
    
    # Sélectionner features dans le bon ordre
    feature_names = model_package['feature_names']
    X = df[feature_names].values
    
    # Normaliser
    X_scaled = model_package['scaler'].transform(X)
    
    return X_scaled

@app.get("/")
async def root():
    # Endpoint de la racine
    return {
        "message": "House Price Prediction API",
        "status": "running",
        "model": model_package['model_name'] if model_package else "not loaded"
    }

@app.get("/health")
async def health():
    # Check santé API
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_package is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: HouseData):
    # Prédiction de prix immobilier
    start_time = time.time()
    
    if not model_package:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Préparer features
        X = prepare_features(data)
        
        # Prédiction
        prediction = model_package['model'].predict(X)[0]
        prediction = max(50000, float(prediction))  # Prix minimum
        
        processing_time = (time.time() - start_time) * 1000
        
        response = PredictionResponse(
            predicted_price=prediction,
            model_name=model_package['model_name'],
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
        # Log audit prédiction
        audit_log = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'prediction',
            'features': data.dict(),
            'prediction': prediction,
            'duration_seconds': processing_time / 1000,
            'model_name': model_package['model_name'],
            'status': 'success'
        }
        logging.info(f"AUDIT: {audit_log}")
        
        return response
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        
        # Log audit erreur
        audit_log = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'prediction_error',
            'features': data.dict(),
            'error': str(e),
            'duration_seconds': processing_time / 1000,
            'status': 'error'
        }
        logging.error(f"AUDIT: {audit_log}")
        
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {e}")

@app.get("/model/info")
async def model_info():
    """Informations sur le modèle"""
    if not model_package:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_name": model_package['model_name'],
        "performance": model_package['performance'],
        "features_count": len(model_package['feature_names']),
        "trained_at": model_package['trained_at']
    }

if __name__ == "__main__":
    import uvicorn
    
    # Charger config (optionnel)
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        host = config['api']['host']
        port = config['api']['port']
    except:
        host = "0.0.0.0"
        port = 8000
    
    logging.info(f"Démarrage API sur {host}:{port}")
    uvicorn.run(app, host=host, port=port)