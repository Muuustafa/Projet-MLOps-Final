FROM python:3.11-slim


WORKDIR /app

# Installer curl pour health check
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers du projet
COPY config.yaml .
COPY train.py .
COPY api.py .
COPY main.py .

# Créer répertoires pour les données et modèles
RUN mkdir -p models data logs

# Port API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande par défaut
CMD ["python", "api.py"]