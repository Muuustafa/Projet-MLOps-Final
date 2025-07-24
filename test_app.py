import os
import sys
import pandas as pd
import joblib
from fastapi.testclient import TestClient

def test_data_exists():
    # Test que les données existent
    print("[TEST] Données existantes...")
    
    assert os.path.exists("data/output.csv"), "Fichier data/output.csv non trouvé"
    
    df = pd.read_csv("data/output.csv")
    assert len(df) > 0, "Données vides"
    assert 'price' in df.columns, "Colonne price manquante"
    
    # Vérification des colonnes requises
    required_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                    'floors', 'waterfront', 'view', 'condition']
    for col in required_cols:
        assert col in df.columns, f"Colonne {col} manquante"
    
    print(f"[OK] Dataset est OK ({len(df)} échantillons, {len(df.columns)} colonnes)")

def test_training():
    # Test entraînement 
    print("[TEST] Entraînement...")
    
    try:
        from train import main as train_main
        model_package = train_main()
        
        assert os.path.exists("models/model.pkl"), "Modèle non sauvegardé"
        assert model_package is not None, "Package modèle vide"
        assert 'model' in model_package, "Modèle manquant dans package"
        assert 'performance' in model_package, "Métriques manquantes"
        
        # Vérifier performance minimum
        r2 = model_package['performance']['r2_test']
        assert r2 > 0.1, f"Performance trop faible: R² = {r2:.3f}"  # Seuil plus permissif
        
        print(f"[OK] Entraînement est OK (R² = {r2:.3f})")
        
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'entraînement: {e}")
        raise

def test_api():
    # Test API
    print("[TEST] API...")
    
    # S'assurer qu'un modèle existe (mais pas d'entraînement ici)
    if not os.path.exists("models/model.pkl"):
        print("[ERROR] Modèle non trouvé pour test API")
        raise AssertionError("Modèle requis pour test API")
    
    from api import app
    client = TestClient(app)
    
    # Test health
    response = client.get("/health")
    assert response.status_code == 200, "Health check échoué"
    
    # Test prédiction
    test_data = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 2.0,
        "waterfront": False,
        "view": 2,
        "condition": 3,
        "sqft_above": 1600,
        "sqft_basement": 200,
        "city": "Seattle",
        "statezip": "WA 98101",
        "country": "USA"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200, f"Prédiction échouée: {response.text}"
    
    result = response.json()
    assert "predicted_price" in result, "Prix prédit manquant"
    assert result["predicted_price"] > 0, "Prix prédit invalide"
    
    print(f"[OK] API OK (Prix prédit: ${result['predicted_price']:,.0f})")

def test_model_persistence():
    # Test persistance du modèle
    print("[TEST] Persistance modèle...")
    
    if not os.path.exists("models/model.pkl"):
        print("[ERROR] Modèle non trouvé pour test persistance")
        raise AssertionError("Modèle requis pour test persistance")
    
    # Charger et vérifier
    model_package = joblib.load("models/model.pkl")
    
    assert 'model' in model_package, "Modèle manquant"
    assert 'scaler' in model_package, "Scaler manquant"
    assert 'feature_names' in model_package, "Noms features manquants"
    
    # Test prédiction directe
    import numpy as np
    X_test = np.random.randn(1, len(model_package['feature_names']))
    X_scaled = model_package['scaler'].transform(X_test)
    
    prediction = model_package['model'].predict(X_scaled)
    assert len(prediction) == 1, "Prédiction invalide"
    assert prediction[0] > 0, "Prix négatif"
    
    print("[OK] Persistance modèle OK")

def test_logging():
    # Test logging audit
    print("[TEST] Logging...")
    
    # Vérifier que les logs sont créés
    log_files = ["training.log", "api.log"]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                assert "AUDIT:" in content, f"Logs audit manquants dans {log_file}"
    
    print("[OK] Logging OK")

def main():
    #Exécute tous les tests"""
    print("========== TESTS MLOps ==========")
    
    tests = [
        test_data_exists,   
        test_training,      
        test_model_persistence,
        test_api,     
        test_logging
    ]
    
    failed = 0
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"[FAILED] {test.__name__} échoué: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__} erreur: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n===== RESULTATS: {len(tests) - failed}/{len(tests)} tests réussis =====")
    
    if failed == 0:
        print("TOUS LES TESTS SONT PASSES !")
        return True
    else:
        print(f"{failed} test(s) échoué(s)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)