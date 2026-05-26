FROM python:3.11-slim

# System libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY pyproject.toml ./
COPY qci/__init__.py qci/__init__.py
RUN pip install --no-cache-dir -e ".[server]" 2>/dev/null || \
    pip install --no-cache-dir \
        "fastapi>=0.104" "uvicorn[standard]>=0.24" \
        "python-multipart>=0.0.6" "httpx>=0.25" \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        && pip install --no-cache-dir -e .

# Copy application code
COPY qci/ qci/
COPY configs/ configs/
COPY scripts/ scripts/
COPY frontend/ frontend/

RUN mkdir -p results

EXPOSE 8000

CMD ["uvicorn", "qci.server.api:app", "--host", "0.0.0.0", "--port", "8000"]
