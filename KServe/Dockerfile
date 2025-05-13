# Use the official Python image from the slim-buster variant
FROM python:3.11.0-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files to the container
COPY pyproject.toml poetry.lock ./
COPY model.joblib ./
COPY crop_label_encoder.joblib ./
COPY risk_label_encoder.joblib ./
COPY scaler.joblib ./
COPY inference.py ./

# Expose the port the app runs on
EXPOSE 8080

# Install Poetry and dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Command to run the application
CMD ["python", "inference.py"]
