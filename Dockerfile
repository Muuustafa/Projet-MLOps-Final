FROM python:3.11-slim

WORKDIR /app

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste
COPY . .

# Créer dossiers
RUN mkdir -p models logs

# Port
EXPOSE 8000

# Démarrer l'API
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]