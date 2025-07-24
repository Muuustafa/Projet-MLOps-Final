import subprocess
import sys
import os
import time

def check_model_exists():
    """Vérifie si le modèle existe"""
    if os.path.exists("models/model.pkl"):
        try:
            import joblib
            model_package = joblib.load("models/model.pkl")
            print(f"Modèle trouvé: {model_package.get('model_name', 'Unknown')}")
            return True
        except Exception as e:
            print(f"Erreur chargement modèle: {e}")
            return False
    else:
        print("Modèle non trouvé dans models/model.pkl")
        return False

def check_api_running():
    """Vérifie si l'API est accessible (optionnel)"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def train_model_if_needed():
    """Entraîne le modèle s'il n'existe pas"""
    if not os.path.exists("models/model.pkl"):
        print("Modèle manquant, tentative d'entraînement...")
        try:
            os.makedirs("models", exist_ok=True)
            
            # Vérifier si train.py existe
            if not os.path.exists("train.py"):
                print("ERREUR: train.py non trouvé")
                return False
                
            # Vérifier si les données existent
            if not os.path.exists("data/output.csv"):
                print("ERREUR: data/output.csv non trouvé")
                return False
            
            print("Lancement de l'entraînement...")
            result = subprocess.run([sys.executable, "train.py"], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("Entraînement réussi")
                return True
            else:
                print(f"Entraînement échoué: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Entraînement timeout (5 minutes)")
            return False
        except Exception as e:
            print(f"Erreur entraînement: {e}")
            return False
    
    return True

def launch_web_interface():
    """Lance l'interface web avec modèle direct"""
    print("=" * 60)
    print("INTERFACE WEB MLOps - MODE MODELE DIRECT")
    print("=" * 60)
    
    # Étape 1: Vérifier et préparer le modèle
    print("Étape 1: Vérification du modèle...")
    
    if not check_model_exists():
        print("Tentative d'entraînement du modèle...")
        if not train_model_if_needed():
            print("ERREUR: Impossible de créer/charger le modèle")
            print("\nSolutions:")
            print("1. Vérifiez que data/output.csv existe")
            print("2. Lancez manuellement: python train.py")
            print("3. Vérifiez les logs d'erreur")
            return
        
        if not check_model_exists():
            print("ERREUR: Modèle toujours non disponible après entraînement")
            return
    
    print("✓ Modèle disponible pour Streamlit")
    
    # Étape 2: Vérifier l'API (optionnel)
    print("\nÉtape 2: Vérification API (optionnelle)...")
    api_available = check_api_running()
    if api_available:
        print("✓ API également disponible sur http://localhost:8000")
    else:
        print("- API non disponible (ce n'est pas grave, on utilise le modèle direct)")
    
    # Étape 3: Lancer Streamlit
    print("\nÉtape 3: Lancement de Streamlit...")
    
    # Vérifier les dépendances
    try:
        import streamlit
        print("✓ Streamlit installé")
    except ImportError:
        print("ERREUR: Streamlit non installé")
        print("Installez avec: pip install streamlit plotly")
        return
    
    # Vérifier le fichier web_ui.py
    if not os.path.exists("web_ui.py"):
        print("ERREUR: Fichier web_ui.py non trouvé")
        print("Créez le fichier web_ui.py dans le répertoire courant")
        return
    
    # Créer un fichier de configuration pour Streamlit
    config_content = f"""
# Configuration automatique pour Streamlit
MODEL_PATH = "models/model.pkl"
API_URL = "http://localhost:8000"
API_AVAILABLE = {api_available}
USE_DIRECT_MODEL = True
"""
    
    with open("streamlit_config.py", "w") as f:
        f.write(config_content)
    
    print("✓ Configuration Streamlit créée")
    
    # Lancer Streamlit
    try:
        print("\n" + "=" * 60)
        print("DÉMARRAGE STREAMLIT")
        print("=" * 60)
        print("🌐 URL: http://localhost:8501")
        print("🛑 Pour arrêter: Ctrl+C")
        print("📊 Mode: Modèle direct + API optionnelle")
        print("-" * 60)
        
        # Variables d'environnement pour Streamlit
        env = os.environ.copy()
        env["STREAMLIT_USE_DIRECT_MODEL"] = "true"
        env["STREAMLIT_MODEL_PATH"] = "models/model.pkl"
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_ui.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], env=env)
        
    except KeyboardInterrupt:
        print("\n✓ Interface fermée par l'utilisateur")
    except Exception as e:
        print(f"✗ Erreur Streamlit: {e}")
        print("Vérifiez l'installation: pip install streamlit plotly")
    finally:
        # Nettoyer le fichier de config
        if os.path.exists("streamlit_config.py"):
            os.remove("streamlit_config.py")

def start_api_only():
    """Lance seulement l'API (fonction séparée)"""
    print("=" * 60)
    print("DÉMARRAGE API SEULEMENT")
    print("=" * 60)
    
    commands_to_try = [
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
        [sys.executable, "api.py"],
        ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
    ]
    
    for cmd in commands_to_try:
        try:
            print(f"Tentative: {' '.join(cmd)}")
            subprocess.run(cmd)
            break
        except Exception as e:
            print(f"Échec: {e}")
            continue

def show_menu():
    """Affiche le menu de choix"""
    print("=" * 60)
    print("LANCEUR MLOps")
    print("=" * 60)
    print("1. Interface Web (modèle direct) - RECOMMANDÉ")
    print("2. API seulement")
    print("3. Vérifier le modèle")
    print("4. Entraîner le modèle")
    print("5. Quitter")
    print("-" * 60)
    
    choice = input("Votre choix (1-5): ").strip()
    
    if choice == "1":
        launch_web_interface()
    elif choice == "2":
        start_api_only()
    elif choice == "3":
        if check_model_exists():
            print("✓ Modèle OK")
        else:
            print("✗ Problème avec le modèle")
    elif choice == "4":
        train_model_if_needed()
    elif choice == "5":
        print("Au revoir!")
        sys.exit(0)
    else:
        print("Choix invalide")
        show_menu()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--web":
            launch_web_interface()
        elif sys.argv[1] == "--api":
            start_api_only()
        else:
            print("Usage: python launcher.py [--web|--api]")
    else:
        show_menu()