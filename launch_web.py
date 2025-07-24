import subprocess
import sys
import os
import time
import threading

def check_api_running():
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_api_background():
    print("🚀 Démarrage de l'API en arrière-plan...")
    try:
        # Démarrer l'API sans afficher la sortie
        subprocess.Popen([
            sys.executable, "main.py", "api"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Attendre que l'API soit prête
        for i in range(30):  # 30 secondes max
            if check_api_running():
                print("✅ API prête !")
                return True
            time.sleep(1)
            if i % 5 == 0:
                print(f"⏳ Attente API... ({i+1}/30s)")
        
        print("⚠️ API lente à démarrer, continuons quand même")
        return False
        
    except Exception as e:
        print(f"❌ Erreur démarrage API: {e}")
        return False

def launch_web_interface():
    
    print("🌐 Lancement de l'interface web MLOps...")
    
    # Vérifier si l'API tourne
    if not check_api_running():
        print("🔌 API non détectée, démarrage automatique...")
        start_api_background()
        time.sleep(3)  # Laisser le temps à l'API de démarrer
    else:
        print("✅ API déjà en cours d'exécution")
    
    # Lancer Streamlit
    try:
        print("🎨 Ouverture de l'interface web...")
        print("📍 URL: http://localhost:8501")
        print("⏹️  Arrêt: Ctrl+C")
        print("-" * 50)
        
        # Lancer streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_ui.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Interface fermée par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur Streamlit: {e}")
        print("💡 Essayez: pip install streamlit plotly")

if __name__ == "__main__":
    launch_web_interface()