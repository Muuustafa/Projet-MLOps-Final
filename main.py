import sys
import os
import subprocess
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train():
    logging.info("Entraînement du modèle...")
    from train import main as train_main
    train_main()
    logging.info("Entraînement terminé")

def start_api():
    logging.info("Démarrage de l'API...")
    from api import app
    import uvicorn
    
    # Lire la config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    host = config['api']['host']
    port = config['api']['port']
    
    logging.info(f"API sur {host}:{port}")
    uvicorn.run(app, host=host, port=port)

def start_web():
    logging.info("Démarrage interface web...")
    
    try:
        import streamlit
    except ImportError:
        logging.error("Streamlit non installé")
        logging.info("Installez avec: pip install streamlit plotly requests")
        return
    
    # Vérifier si l'API tourne, sinon la démarrer
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code == 200:
            logging.info("API déjà active")
        else:
            logging.info("Démarrage de l'API en arrière-plan...")
            # Démarrer l'API en arrière-plan
            subprocess.Popen([sys.executable, "main.py", "api"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(5)
    except:
        logging.info("Démarrage de l'API en arrière-plan...")
        subprocess.Popen([sys.executable, "main.py", "api"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time
        time.sleep(5)
    
    # Lancer Streamlit
    try:
        logging.info("Ouverture de l'interface web...")
        logging.info("URL: http://localhost:8501")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_ui.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except Exception as e:
        logging.error(f"Erreur Streamlit: {e}")

def run_tests():
    logging.info("Exécution des tests...")
    
    try:
        # Importer et exécuter directement
        from test_app import main as test_main
        success = test_main()
        
        if success:
            logging.info("Tests réussis")
        else:
            logging.error("Tests échoués")
        
        return success
        
    except Exception as e:
        logging.error(f"Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all():
    logging.info("Pipeline MLOps complet...")
    
    # 1. Tests
    if not run_tests():
        logging.error("Arrêt: tests échoués")
        return False
    
    # 2. Entraînement
    train()
    
    # 3. Validation que le modèle fonctionne
    if os.path.exists('models/model.pkl'):
        logging.info("Pipeline terminé avec succès")
        logging.info("Démarrez l'API avec: python main.py api")
        logging.info("Ou l'interface web avec: python main.py web")
        return True
    else:
        logging.error("Échec pipeline: modèle non créé")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py [train|api|web|test|all]")
        print("  train - Entraîne le modèle")
        print("  api   - Démarre l'API REST")
        print("  web   - Lance l'interface web moderne")
        print("  test  - Exécute les tests")
        print("  all   - Pipeline complet")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "train":
            train()
        elif command == "api":
            start_api()
        elif command == "web":
            start_web()
        elif command == "test":
            success = run_tests()
            sys.exit(0 if success else 1)
        elif command == "all":
            success = run_all()
            sys.exit(0 if success else 1)
        else:
            print(f"Commande inconnue: {command}")
            print("Commandes disponibles: train, api, web, test, all")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Arrêt par utilisateur")
    except Exception as e:
        logging.error(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()