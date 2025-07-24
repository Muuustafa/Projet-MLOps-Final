import subprocess
import sys
import os
import time
import threading

def check_api_running():
    """Vérifie si l'API est accessible"""
    try:
        import requests
        print("Test de connexion à l'API...")
        response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"Réponse API: {response.status_code}")
        return response.status_code == 200
    except ImportError:
        print("ERREUR: module 'requests' non installé. Installez avec: pip install requests")
        return False
    except requests.exceptions.ConnectionError:
        print("API non accessible sur http://localhost:8000")
        return False
    except requests.exceptions.Timeout:
        print("Timeout lors de la connexion à l'API")
        return False
    except Exception as e:
        print(f"Erreur lors du test API: {e}")
        return False

def check_api_process():
    """Vérifie si un processus API tourne déjà"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'main.py' in cmdline and 'api' in cmdline:
                    print(f"Processus API trouvé: PID {proc.info['pid']}")
                    return True
                if 'uvicorn' in cmdline and ('main:app' in cmdline or 'api:app' in cmdline):
                    print(f"Processus uvicorn trouvé: PID {proc.info['pid']}")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    except ImportError:
        print("Module psutil non disponible, utilisation méthode basique")
        return False

def start_api_background():
    """Démarre l'API en arrière-plan"""
    print("Démarrage de l'API en arrière-plan...")
    
    try:
        # Plusieurs tentatives de commandes possibles
        commands_to_try = [
            [sys.executable, "main.py", "api"],
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
            ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
        ]
        
        process = None
        for cmd in commands_to_try:
            try:
                print(f"Tentative commande: {' '.join(cmd)}")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                time.sleep(2)  # Laisser le temps de démarrer
                
                # Vérifier si le processus est encore vivant
                if process.poll() is None:
                    print("Processus démarré avec succès")
                    break
                else:
                    stdout, stderr = process.communicate()
                    print(f"Commande échouée. Stdout: {stdout[:200]}... Stderr: {stderr[:200]}...")
                    process = None
            except FileNotFoundError:
                print(f"Commande non trouvée: {cmd[0]}")
                continue
            except Exception as e:
                print(f"Erreur avec commande {cmd[0]}: {e}")
                continue
        
        if process is None:
            print("ERREUR: Impossible de démarrer l'API avec aucune commande")
            return False
        
        # Attendre que l'API soit prête
        print("Attente que l'API soit prête...")
        for i in range(60):  # 60 secondes max
            if check_api_running():
                print("API prête et accessible!")
                return True
            time.sleep(1)
            if i % 10 == 0 and i > 0:
                print(f"Attente API... ({i}/60s)")
                # Vérifier si le processus est encore vivant
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    print(f"Processus API s'est arrêté. Stdout: {stdout[-500:]}")
                    print(f"Stderr: {stderr[-500:]}")
                    return False
        
        print("ATTENTION: API lente à répondre, mais processus actif")
        return True
        
    except Exception as e:
        print(f"Erreur démarrage API: {e}")
        return False

def test_manual_commands():
    """Affiche les commandes à tester manuellement"""
    print("\nCommandes à tester manuellement:")
    print("1. python main.py api")
    print("2. uvicorn main:app --host 0.0.0.0 --port 8000")
    print("3. uvicorn api:app --host 0.0.0.0 --port 8000")
    print("4. python -m uvicorn main:app --host 0.0.0.0 --port 8000")
    print("\nTestez chaque commande dans un terminal séparé")
    print("Puis relancez ce script")

def launch_web_interface():
    """Lance l'interface web MLOps"""
    print("=" * 60)
    print("LANCEMENT DE L'INTERFACE WEB MLOps")
    print("=" * 60)
    
    # Vérifier d'abord si l'API répond
    print("Étape 1: Vérification de l'API...")
    if check_api_running():
        print("API déjà accessible sur http://localhost:8000")
    else:
        print("API non accessible")
        
        # Vérifier si un processus API tourne
        print("Étape 2: Recherche de processus API...")
        if check_api_process():
            print("Processus API détecté mais non accessible")
            print("L'API met peut-être du temps à démarrer...")
            
            # Attendre un peu plus
            for i in range(30):
                if check_api_running():
                    print("API maintenant accessible!")
                    break
                time.sleep(2)
                if i % 5 == 0:
                    print(f"Attente... ({i*2}/60s)")
            else:
                print("API toujours non accessible après attente")
                test_manual_commands()
                return
        else:
            print("Aucun processus API détecté")
            print("Étape 3: Tentative de démarrage automatique...")
            
            if not start_api_background():
                print("ECHEC du démarrage automatique")
                test_manual_commands()
                return
    
    # Vérification finale
    if not check_api_running():
        print("ERREUR: API toujours non accessible")
        print("Veuillez démarrer l'API manuellement:")
        test_manual_commands()
        return
    
    # Lancer Streamlit
    try:
        print("\n" + "=" * 60)
        print("LANCEMENT DE STREAMLIT")
        print("=" * 60)
        print("URL: http://localhost:8501")
        print("Pour arrêter: Ctrl+C")
        print("-" * 60)
        
        # Vérifier si streamlit est installé
        try:
            import streamlit
        except ImportError:
            print("ERREUR: Streamlit non installé")
            print("Installez avec: pip install streamlit plotly")
            return
        
        # Vérifier si le fichier web_ui.py existe
        if not os.path.exists("web_ui.py"):
            print("ERREUR: Fichier web_ui.py non trouvé")
            print("Assurez-vous que web_ui.py est dans le répertoire courant")
            return
        
        # Lancer streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_ui.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\nInterface fermée par l'utilisateur")
    except Exception as e:
        print(f"Erreur Streamlit: {e}")
        print("Vérifiez l'installation: pip install streamlit plotly")

if __name__ == "__main__":
    launch_web_interface()