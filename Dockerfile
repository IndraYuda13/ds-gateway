FROM node:20-slim AS node-base

FROM python:3.11-slim

# Install Node.js runtime from node-base for sub-100ms WASM PoW execution
COPY --from=node-base /usr/local/bin/node /usr/local/bin/node

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8550
ENV HOST=0.0.0.0
ENV DATA_DIR=/app/data

EXPOSE 8550

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8550/health || exit 1

CMD ["python", "main.py"]
