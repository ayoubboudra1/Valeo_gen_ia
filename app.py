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

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()
app = Flask(__name__)

DATA_PATH = os.getenv("DATA_PATH") or "./documents"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 30)
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL") or "sentence-transformers/all-mpnet-base-v2"
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL") or "http://localhost:9200"
INDEX_NAME = os.getenv("INDEX_NAME") or "document_index"
NUM_RESULTS = int(os.getenv("NUM_RESULTS") or 3)

def load_docs(directory: str):
    loader = DirectoryLoader(directory, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def initialize_app():
    documents = load_docs(DATA_PATH)
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
    docs = split_docs(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vector_store = ElasticsearchStore.from_documents(
        documents=docs,
        embedding=embeddings,
        es_url=ELASTICSEARCH_URL,
        index_name=INDEX_NAME,
    )
    print(f"Vector store created in Elasticsearch at index '{INDEX_NAME}'")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.1,
        device_map="auto",
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={'k': NUM_RESULTS}),
        memory=memory,
        verbose=True
    )
    return chain

chain = initialize_app()

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        result = chain({"question": question})
        answer = result["answer"]
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)