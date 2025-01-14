from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import ElasticsearchStore
import torch
import os
import warnings
from dotenv import load_dotenv

# Ignorer les avertissements de dépréciation pour éviter les messages inutiles dans la console
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Charger les variables d'environnement depuis un fichier .env
load_dotenv()

# Initialiser l'application Flask
app = Flask(__name__)

# Configuration des variables d'environnement avec des valeurs par défaut
DATA_PATH = os.getenv("DATA_PATH") or "./documents"  # Chemin vers les documents
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512)  # Taille des morceaux de texte
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 30)  # Chevauchement entre les morceaux
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL") or "sentence-transformers/all-mpnet-base-v2"  # Modèle d'embedding
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL") or "http://localhost:9200"  # URL d'Elasticsearch
INDEX_NAME = os.getenv("INDEX_NAME") or "document_index"  # Nom de l'index Elasticsearch
NUM_RESULTS = int(os.getenv("NUM_RESULTS") or 3)  # Nombre de résultats à retourner

# Fonction pour charger les documents PDF depuis un répertoire
def load_docs(directory: str):
    loader = DirectoryLoader(directory, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Fonction pour diviser les documents en morceaux de texte
def split_docs(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Fonction pour initialiser l'application
def initialize_app():
    # Charger les documents depuis le répertoire spécifié
    documents = load_docs(DATA_PATH)
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")

    # Diviser les documents en morceaux de texte
    docs = split_docs(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Initialiser le modèle d'embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # Créer un stockage vectoriel dans Elasticsearch
    vector_store = ElasticsearchStore.from_documents(
        documents=docs,
        embedding=embeddings,
        es_url=ELASTICSEARCH_URL,
        index_name=INDEX_NAME,
    )
    print(f"Vector store created in Elasticsearch at index '{INDEX_NAME}'")

    # Charger le modèle de génération de texte (GPT-2)
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    # Configurer le pipeline de génération de texte
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.1,
        device_map="auto",
    )

    # Initialiser le modèle de langage avec le pipeline
    llm = HuggingFacePipeline(pipeline=pipe)

    # Configurer la mémoire pour garder l'historique de la conversation
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Créer une chaîne de récupération conversationnelle
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={'k': NUM_RESULTS}),
        memory=memory,
        verbose=True
    )
    return chain

# Initialiser la chaîne de traitement
chain = initialize_app()

# Route pour interroger le système avec une question
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")

    # Vérifier si une question a été fournie
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Obtenir la réponse à la question
        result = chain({"question": question})
        answer = result["answer"]
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        # Gérer les erreurs et retourner un message d'erreur
        return jsonify({"error": str(e)}), 500

# Route pour vérifier l'état de santé de l'application
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

# Démarrer l'application Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)