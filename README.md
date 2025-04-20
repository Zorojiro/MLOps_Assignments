# Bank Marketing ML Inference Web Application

A production-level machine learning inference web application for predicting customer subscription to a term deposit based on bank marketing data. The application is containerized with Docker and deployed using Kubernetes, KServe, and Kubeflow with CI/CD automation via GitHub Actions.

## Project Overview

This project implements an end-to-end machine learning solution for bank marketing predictions with the following components:

1. **Machine Learning Model**: A RandomForest classifier to predict if a customer will subscribe to a term deposit.
2. **Web Interface**: A Flask web application that provides a user-friendly interface for making predictions.
3. **API Endpoints**: RESTful API endpoints for single and batch predictions.
4. **Containerization**: Docker configuration for packaging the application.
5. **Kubernetes Deployment**: Resources for deploying on Kubernetes clusters.
6. **Model Serving**: KServe for efficient model serving in production.
7. **ML Pipelines**: Kubeflow pipelines for orchestrating the ML workflow.
8. **CI/CD**: GitHub Actions workflows for continuous integration and deployment.

## Project Structure

```
.
├── app.py                      # Flask web application
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── .github/
│   └── workflows/
│       ├── train.yaml          # GitHub Action for model training
│       └── deploy.yaml         # GitHub Action for deployment
├── kube/
│   ├── main.py                 # Model training script
│   ├── inference.py            # Model inference script
│   └── data/                   # Dataset directory
├── kserve/
│   ├── inferenceservice.yaml   # KServe configuration
│   └── model.py                # KServe model handler
├── kubernetes/
│   ├── deployment.yaml         # Kubernetes deployment
│   ├── service.yaml            # Kubernetes service
│   └── ingress.yaml            # Kubernetes ingress
├── kubeflow/
│   ├── pipeline.py             # Kubeflow pipeline definition
│   └── workflow.yaml           # Argo workflow template
└── templates/
    └── index.html              # Web interface template
```

## Features

- **Interactive UI**: User-friendly web interface for inputting customer data and viewing predictions.
- **Real-time Predictions**: Instant predictions using the trained ML model.
- **Batch Processing**: Support for processing multiple records via CSV upload.
- **Monitoring & Metrics**: Prediction probabilities and confidence scores.
- **Scalable Architecture**: Kubernetes-based deployment for horizontal scaling.
- **Model Versioning**: Automated model training and deployment pipeline.
- **High Availability**: Multiple replicas for fault tolerance.

## Technical Implementation

### ML Model

- **Algorithm**: Random Forest Classifier
- **Features**: Customer demographics, contact information, campaign details, and economic indicators
- **Target**: Binary classification (yes/no) for term deposit subscription
- **Pipeline**: Data preprocessing, feature engineering, model training, and evaluation

### Web Application

- **Backend**: Flask REST API
- **Frontend**: HTML, CSS (Bootstrap), JavaScript (jQuery)
- **Endpoints**:
  - `/` - Main web interface
  - `/predict` - Single prediction API endpoint
  - `/batch-predict` - Batch prediction endpoint

### Deployment Pipeline

1. **Model Training**:
   - Triggered weekly or manually
   - Trains the model on the latest data
   - Uploads the model to S3 storage

2. **Deployment**:
   - Triggered by code changes or successful model training
   - Builds and pushes Docker image
   - Updates Kubernetes deployments
   - Deploys KServe InferenceService

### Kubeflow Components

- **Data Preparation**: Splits data into training and testing sets
- **Model Training**: Trains and saves the ML model
- **Model Evaluation**: Evaluates model performance and generates metrics
- **Deployment**: Conditionally deploys the model if it meets quality thresholds

## Prerequisites

- Docker
- Kubernetes cluster with KServe and Kubeflow installed
- S3-compatible storage (for model artifacts)
- GitHub repository (for CI/CD)

## Setup & Deployment

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Access the web interface at `http://localhost:5000`

### Containerized Deployment

1. Build the Docker image:
   ```bash
   docker build -t bank-marketing-app:latest .
   ```

2. Run the container:
   ```bash
   docker run -p 5000:5000 bank-marketing-app:latest
   ```

### Kubernetes Deployment

1. Apply Kubernetes manifests:
   ```bash
   kubectl apply -f kubernetes/deployment.yaml
   kubectl apply -f kubernetes/service.yaml
   kubectl apply -f kubernetes/ingress.yaml
   ```

2. Deploy the KServe InferenceService:
   ```bash
   kubectl apply -f kserve/inferenceservice.yaml
   ```

### Kubeflow Pipeline

1. Upload the Kubeflow pipeline:
   ```bash
   python kubeflow/pipeline.py
   ```

2. Submit the Argo workflow:
   ```bash
   kubectl create -f kubeflow/workflow.yaml
   ```

## CI/CD Configuration

The following GitHub secrets need to be configured:

- `DOCKERHUB_USERNAME` - Docker Hub username
- `DOCKERHUB_TOKEN` - Docker Hub access token
- `AWS_ACCESS_KEY_ID` - AWS access key for S3
- `AWS_SECRET_ACCESS_KEY` - AWS secret key for S3
- `KUBE_CONFIG` - Base64-encoded Kubernetes configuration
- `SLACK_WEBHOOK_URL` - Slack webhook for notifications

## Monitoring & Scaling

- **Horizontal Pod Autoscaler**: Scales based on CPU/memory usage
- **Liveness/Readiness Probes**: Ensures application health
- **Resource Requests/Limits**: Proper resource allocation

## Future Improvements

- [ ] Implement A/B testing for model variants
- [ ] Add model explainability features
- [ ] Integrate with MLflow for experiment tracking
- [ ] Set up Prometheus & Grafana for monitoring
- [ ] Implement canary deployments

## License

MIT

## Contributors

- Your Name