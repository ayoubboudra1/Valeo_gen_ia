# Utiliser une image de base Python 3.12.7 slim pour réduire la taille de l'image
FROM python:3.12.7-slim

# Définir les variables d'environnement pour configurer l'application
# URL d'Elasticsearch
ENV ELASTICSEARCH_URL=http://elasticsearch:9200 
# Nom de l'index Elasticsearch 
ENV INDEX_NAME=document_index  
# Chemin vers les documents dans le conteneur
ENV DATA_PATH=/app/documents  
# Taille des morceaux de texte
ENV CHUNK_SIZE=512
# Chevauchement entre les morceaux de texte  
ENV CHUNK_OVERLAP=30  
# Modèle d'embedding
ENV EMBEDDINGS_MODEL=sentence-transformers/all-mpnet-base-v2 
# Nombre de résultats à retourner
ENV NUM_RESULTS=3  

# Mettre à jour les paquets et installer curl pour vérifier la disponibilité d'Elasticsearch
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*  # Nettoyer le cache pour réduire la taille de l'image

# Mettre à jour pip pour installer les dernières versions des paquets
RUN pip install --upgrade pip

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt /app/requirements.txt

# Installer les dépendances Python listées dans requirements.txt
RUN pip install -r /app/requirements.txt

# Copier le dossier des documents dans le conteneur
COPY documents /app/documents

# Copier tout le contenu du répertoire courant dans le conteneur
COPY . /app

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Exposer le port 5000 pour permettre l'accès à l'application Flask
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["sh", "-c", "echo 'Waiting for Elasticsearch to start...' && \
    until curl -s ${ELASTICSEARCH_URL} >/dev/null; do sleep 1; done && \
    echo 'Elasticsearch is up!' && \
    python app.py"]