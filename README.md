# Assistant Documentaire avec RAG (Retrieval-Augmented Generation)

Ce projet est une application Flask qui utilise un système de Retrieval-Augmented Generation (RAG) pour répondre à des questions basées sur des documents PDF. L'application est conteneurisée avec Docker et utilise Elasticsearch pour le stockage et la recherche de documents.

## Structure des fichiers

.
├── documents/ # Dossier contenant les documents PDF
├── env # Fichier de variables d'environnement
├── env.example # Exemple de fichier de variables d'environnement
├── .gitignore # Fichier pour ignorer certains fichiers dans Git
├── app.py # Script principal de l'application Flask
├── docker-compose.yml # Fichier pour démarrer Elasticsearch et l'application
├── Dockerfile # Fichier pour construire l'image Docker de l'application
├── list_of_question_to_test.txt # Liste de questions pour tester l'application
├── README.md # Ce fichier
├── requirements.txt # Dépendances Python nécessaires

## Prérequis

- **Docker** : Assurez-vous d'avoir Docker installé sur votre machine.
- **Docker Compose** : Nécessaire pour démarrer Elasticsearch et l'application ensemble.
- **Fichiers PDF** : Placez les documents PDF dans le dossier `documents/`.

## Configuration

1. **Variables d'environnement** :

   - Copiez le fichier `env.example` vers `.env` et modifiez les valeurs si nécessaire.
   - Exemple de `.env` :
     ```plaintext
     DATA_PATH=./documents
     CHUNK_SIZE=512
     CHUNK_OVERLAP=30
     EMBEDDINGS_MODEL=sentence-transformers/all-mpnet-base-v2
     ELASTICSEARCH_URL=http://elasticsearch:9200
     INDEX_NAME=document_index
     NUM_RESULTS=3
     ```

2. **Dépendances** :
   - Les dépendances Python sont listées dans `requirements.txt`. Elles seront installées automatiquement lors de la construction de l'image Docker.

## Utilisation

### 1. Construire et démarrer les conteneurs

Pour démarrer l'application et Elasticsearch, utilisez Docker Compose :

```bash
docker-compose up --build
```

Cela va :

- Construire l'image Docker de l'application.

- Démarrer un conteneur Elasticsearch.

- Démarrer l'application Flask.

### 2. Accéder à l'application

Une fois les conteneurs démarrés, l'application Flask sera accessible à l'adresse suivante :

```bash
http://localhost:5000
```

### 3. Poser une question

Vous pouvez interroger l'application en envoyant une requête POST à l'endpoint /query avec une question au format JSON.

Exemple de requête :

```bash
curl -X POST http://localhost:5000/query \
    -H "Content-Type: application/json" \
    -d '{"question": "Quelle est la politique de confidentialité ?"}'
```

Réponse attendue :

```bash
{
    "question": "Quelle est la politique de confidentialité ?",
    "answer": "La politique de confidentialité stipule que..."
}
```

### 4. Vérifier l'état de santé

Pour vérifier si l'application fonctionne correctement, vous pouvez accéder à l'endpoint `/health` :

```bash
curl http://localhost:5000/health
```

Réponse attendue :

```bash
{
    "status": "healthy"
}
```

### 5. Arrêter les conteneurs

Pour arrêter les conteneurs, utilisez la commande suivante :

```bash
docker-compose down
```

### Tests

Vous pouvez tester l'application avec une liste de questions prédéfinies dans le fichier `list_of_question_to_test.txt`. Exécutez simplement les questions via des requêtes POST comme indiqué ci-dessus.

### Personnalisation

- **Documents** : Ajoutez ou modifiez les fichiers PDF dans le dossier `documents/`.
- **Modèle d'embedding** : Changez le modèle d'embedding dans le fichier `.env` si nécessaire.
- **Configuration Elasticsearch** : Modifiez l'URL ou l'index dans le fichier `.env`.

### Remarques

- Assurez-vous qu'Elasticsearch est bien démarré avant que l'application Flask ne commence à traiter les requêtes.
