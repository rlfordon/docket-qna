FROM python:3.11-slim

WORKDIR /app

# Install dependencies (no torch/sentence-transformers needed — using OpenAI embeddings)
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

# Copy application code
COPY . .

# Copy pre-built demo data into the runtime data directory
RUN cp -r data-demo/cases data/cases && cp -r data-demo/chroma data/chroma

EXPOSE 10000

CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]
