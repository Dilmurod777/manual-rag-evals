FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app eval public README.md ./

# The manual and its embeddings live in data/, built at start when absent.
ENV OLLAMA_HOST=http://host.docker.internal:11434
EXPOSE 8000
CMD ["sh", "-c", "test -f data/embeddings.npz || python -m app.setup; exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
