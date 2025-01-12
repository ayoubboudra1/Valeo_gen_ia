FROM python:3.12.7-slim

ENV ELASTICSEARCH_URL=http://elasticsearch:9200
ENV INDEX_NAME=document_index
ENV DATA_PATH=/app/documents
ENV CHUNK_SIZE=512
ENV CHUNK_OVERLAP=30
ENV EMBEDDINGS_MODEL=sentence-transformers/all-mpnet-base-v2
ENV NUM_RESULTS=3

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt


COPY documents /app/documents

COPY . /app
WORKDIR /app

EXPOSE 5000

CMD ["sh", "-c", "echo 'Waiting for Elasticsearch to start...' && \
    until curl -s ${ELASTICSEARCH_URL} >/dev/null; do sleep 1; done && \
    echo 'Elasticsearch is up!' && \
    python app.py"]