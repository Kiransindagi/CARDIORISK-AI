FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY src/ src/
COPY data/ data/
COPY artifacts/ artifacts/

# Expose backend port
EXPOSE 8000

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Run production server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
