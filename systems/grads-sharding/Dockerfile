FROM python:3.10-slim

WORKDIR /app

# Copy shared code
COPY shared/ /app/shared/

# Copy GradsSharding implementation
COPY grads-sharding/ /app/grads_sharding/

# Install dependencies
RUN pip install --no-cache-dir -r /app/shared/requirements.txt

# Set Python path
ENV PYTHONPATH=/app

# Default entrypoint for GradsSharding
ENTRYPOINT ["python", "-m", "grads_sharding.run"]
