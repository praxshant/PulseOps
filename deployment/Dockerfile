FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Create volume mount points
RUN mkdir -p /app/artifacts/runs /app/artifacts/inference

# Copy and setup healthcheck script
COPY deployment/scripts/healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD /healthcheck.sh

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
