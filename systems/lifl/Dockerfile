FROM python:3.10-slim

WORKDIR /app

# Copy shared code
COPY shared/ /app/shared/

# Copy LIFL implementation
COPY lifl/ /app/lifl/

# Install dependencies
RUN pip install --no-cache-dir -r /app/shared/requirements.txt

# Set Python path
ENV PYTHONPATH=/app

# Default entrypoint for LIFL
ENTRYPOINT ["python", "-m", "lifl.run"]
