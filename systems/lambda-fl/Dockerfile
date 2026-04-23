FROM python:3.10-slim

WORKDIR /app

# Copy shared code
COPY shared/ /app/shared/

# Copy Lambda-FL implementation
COPY lambda-fl/ /app/lambda_fl/

# Install dependencies
RUN pip install --no-cache-dir -r /app/shared/requirements.txt

# Set Python path
ENV PYTHONPATH=/app

# Default entrypoint for Lambda-FL
ENTRYPOINT ["python", "-m", "lambda_fl.run"]
