FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour le parsing PDF (unstructured/pdfminer) et le build de sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ChromaDB et docs_txt doivent être présents (générés par la CI/le DAG d'ingestion)
# ou montés en volume au démarrage du conteneur.
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
