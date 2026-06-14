FROM python:3.11-slim

# Metadata
LABEL maintainer="Reasoning-Agent-Hackathon"
LABEL description="OpenAI Reasoning Multi-Agent System — Microsoft Agent League Hackathon"

WORKDIR /app

# Install OS deps (gcc for some python packages, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py .
COPY run_agent.py .

# Create a non-root user for security
RUN useradd -m -u 1000 agentuser && chown -R agentuser:agentuser /app
USER agentuser

# Force UTF-8 output (fixes Windows-style encoding issues in logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
