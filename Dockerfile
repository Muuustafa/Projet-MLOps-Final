FROM python:3.11-slim

WORKDIR /app

# Copier les requirements d'abord (pour le cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Créer les dossiers nécessaires
RUN mkdir -p data models logs

# Copier les données AVANT le code
COPY data/ ./data/
# Copier le modèle s'il existe déjà
COPY models/ ./models/ 2>/dev/null || true

# Copier le code source
COPY *.py ./
COPY train.py ./
COPY api.py ./
# Copier main.py s'il existe
COPY main.py ./ 2>/dev/null || true

# Exposer le port
EXPOSE 8000

# Variables d'environnement
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8000

# Sanity check - vérifier que les données sont là
RUN ls -la data/ || echo "ATTENTION: Dossier data vide"
RUN test -f data/output.csv && echo "✓ Données trouvées" || echo "⚠️ data/output.csv manquant"

# Commande de démarrage
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]