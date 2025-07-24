
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# On fera tout ici (preprocessing, feature engineering, training, validation)

# Setup logging pour audit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def check_data_exists():
    # On Vérifie que le fichier (data/output.csv) de données existe
    if not os.path.exists("data/output.csv"):
        raise FileNotFoundError(
            "Fichier data/output.csv non trouvé. "
            "Veuillez placer votre dataset dans le répertoire data/"
        )
    logging.info("Dataset trouvé: data/output.csv")

def preprocess_data(df):
    # Un Preprocessing simple et efficace
    df = df.copy()
    
    # Un Feature engineering basique
    df['house_age'] = 2024 - df['yr_built']
    df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
    df['total_sqft'] = df['sqft_living'] + df['sqft_basement']
    
    # Supprimer colonnes inutiles
    drop_cols = ['date', 'street', 'yr_built', 'yr_renovated']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Encoder catégorielles
    cat_cols = ['city', 'statezip', 'country']
    encoders = {}
    
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    # Gérer valeurs manquantes
    df = df.fillna(df.median(numeric_only=True))
    
    return df, encoders

def train_models(X, y):
    # Les modèles que je souhaite entraîner et trouver la meilleure 
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
    }
    
    best_model = None
    best_score = -np.inf
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        logging.info(f"Entraînement {name}...")
        
        # Validation croisée
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        mean_score = cv_scores.mean()
        
        # Entraîner sur toutes les données
        model.fit(X, y)
        
        results[name] = {
            'model': model,
            'cv_score': mean_score,
            'cv_std': cv_scores.std()
        }
        
        logging.info(f"{name} - R^2 CV: {mean_score:.4f} {cv_scores.std():.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name
    
    logging.info(f"Meilleur modèle: {best_name} (R^2 = {best_score:.4f})")
    return best_model, best_name, results

def main():
    start_time = datetime.now()
    logging.info("====== DÉBUT ENTRAÎNEMENT MLOPs ======")
    
    # 1. Vérifier données
    check_data_exists()
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv(config['data']['file'])
    logging.info(f"Données chargées: {df.shape}")
    
    # 2. Preprocessing
    df_processed, encoders = preprocess_data(df)
    
    # 3. Split données
    target = config['data']['target']
    X = df_processed.drop(columns=[target])
    y = df_processed[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entraînement
    best_model, best_name, results = train_models(X_train_scaled, y_train)
    
    # 6. Évaluation finale
    y_pred = best_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logging.info(f"Performance test - R^2: {test_r2:.4f}, RMSE: {test_rmse:.0f}")
    
    # 7. Sauvegarde
    os.makedirs("models", exist_ok=True)
    
    model_package = {
        'model': best_model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': list(X.columns),
        'model_name': best_name,
        'performance': {
            'r2_test': test_r2,
            'rmse_test': test_rmse,
            'cv_scores': {name: res['cv_score'] for name, res in results.items()}
        },
        'trained_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_package, 'models/model.pkl')
    
    # Log structuré pour audit MLOps
    audit_log = {
        'timestamp': datetime.now().isoformat(),
        'event_type': 'model_training',
        'model_name': best_name,
        'dataset_size': len(df),
        'features_count': len(X.columns),
        'performance_r2': test_r2,
        'performance_rmse': test_rmse,
        'training_duration_minutes': (datetime.now() - start_time).total_seconds() / 60
    }
    
    logging.info(f"AUDIT: {audit_log}")
    
    elapsed = datetime.now() - start_time
    logging.info(f"=== ENTRAÎNEMENT TERMINÉ en {elapsed.total_seconds():.1f}s ===")
    
    return model_package

if __name__ == "__main__":
    main()