import kfp
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Artifact, pipeline

# Define the data preparation component
@component(
    packages_to_install=["pandas", "scikit-learn"],
    base_image="python:3.12.9"
)
def prepare_data(
    data_path: str,
    test_size: float,
    output_train: Output[Dataset],
    output_test: Output[Dataset]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import pickle
    
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path, delimiter=';')
    
    # Filter only the 16 specified features plus the target
    feature_columns = [
        'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'poutcome', 'age', 'balance', 'day',
        'duration', 'campaign', 'pdays', 'previous', 'y'
    ]
    
    # Select only the specified features
    data = data[feature_columns]
    
    # Split features and target
    X = data.drop('y', axis=1)
    y = data['y'].map({'yes': 1, 'no': 0})
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Save the datasets
    train_data = pd.concat([X_train, y_train.rename('target')], axis=1)
    test_data = pd.concat([X_test, y_test.rename('target')], axis=1)
    
    train_data.to_pickle(output_train.path)
    test_data.to_pickle(output_test.path)
    
    print(f"Data prepared with {len(train_data)} training samples and {len(test_data)} test samples")
    print(f"Features used: {X_train.columns.tolist()}")

# Define the model training component
@component(
    packages_to_install=["pandas", "scikit-learn", "joblib"],
    base_image="python:3.9"
)
def train_model(
    train_data: Input[Dataset],
    model_output: Output[Model],
    model_metrics: Output[Artifact]
):
    import pandas as pd
    import numpy as np
    import pickle
    import joblib
    import json
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Load the training data
    train_data_df = pd.read_pickle(train_data.path)
    X_train = train_data_df.drop('target', axis=1)
    y_train = train_data_df['target']
    
    # Identify categorical and numerical columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, model_output.path)
    print(f"Model saved to {model_output.path}")
    
    # Record placeholder metrics
    metrics = {
        "framework": "scikit-learn",
        "model_type": "RandomForestClassifier"
    }
    
    with open(model_metrics.path, 'w') as f:
        json.dump(metrics, f)

# Define the model evaluation component
@component(
    packages_to_install=["pandas", "scikit-learn", "joblib", "matplotlib"],
    base_image="python:3.9"
)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    evaluation_output: Output[Artifact],
    evaluation_plots: Output[Artifact]
):
    import pandas as pd
    import numpy as np
    import joblib
    import json
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
    
    # Load the model and test data
    model = joblib.load(model.path)
    test_data_df = pd.read_pickle(test_data.path)
    
    X_test = test_data_df.drop('target', axis=1)
    y_test = test_data_df['target']
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Record metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc)
    }
    
    print("Model Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    with open(evaluation_output.path, 'w') as f:
        json.dump(metrics, f)
    
    # Create evaluation plots
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.yticks([0, 1], ['No', 'Yes'])
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.subplot(2, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Save the plots
    plt.tight_layout()
    plt.savefig(evaluation_plots.path)
    print(f"Evaluation plots saved to {evaluation_plots.path}")

# Define the deployment component
@component(
    packages_to_install=["kubernetes"],
    base_image="python:3.9"
)
def deploy_model(
    model: Input[Model],
    evaluation_metrics: Input[Artifact],
    accuracy_threshold: float,
    kserve_yaml: str
):
    import json
    import os
    import subprocess
    from kubernetes import client, config
    
    # Load the evaluation metrics
    with open(evaluation_metrics.path, 'r') as f:
        metrics = json.load(f)
    
    # Check if the model meets the accuracy threshold
    if metrics["accuracy"] < accuracy_threshold:
        print(f"Model accuracy ({metrics['accuracy']:.4f}) below threshold ({accuracy_threshold}). Skipping deployment.")
        return
    
    print(f"Model accuracy ({metrics['accuracy']:.4f}) meets deployment threshold ({accuracy_threshold}). Proceeding with deployment.")
    
    # Create a temporary file with the KServe yaml
    with open('/tmp/kserve_deployment.yaml', 'w') as f:
        f.write(kserve_yaml)
    
    # Deploy using kubectl
    try:
        subprocess.run(["kubectl", "apply", "-f", "/tmp/kserve_deployment.yaml"], check=True)
        print("Model successfully deployed with KServe")
    except subprocess.CalledProcessError as e:
        print(f"Error deploying model: {e}")
        raise e

# Define the main pipeline
@pipeline(
    name="Bank Marketing ML Pipeline",
    description="End-to-end ML pipeline for bank marketing prediction using 16 specific features"
)
def bank_marketing_pipeline(
    data_path: str = "../kube/data/bank-full.csv",
    test_size: float = 0.2,
    accuracy_threshold: float = 0.8
):
    # Prepare the data
    data_prep_task = prepare_data(data_path=data_path, test_size=test_size)
    
    # Train the model
    training_task = train_model(
        train_data=data_prep_task.outputs["output_train"]
    )
    
    # Evaluate the model
    evaluation_task = evaluate_model(
        model=training_task.outputs["model_output"],
        test_data=data_prep_task.outputs["output_test"]
    )
    
    # Define the KServe YAML
    kserve_yaml = """
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "bank-marketing-model"
spec:
  predictor:
    sklearn:
      storageUri: "s3://mlops-models/bank-marketing/"
      resources:
        requests:
          cpu: "100m"
          memory: "256Mi"
        limits:
          cpu: "500m"
          memory: "512Mi"
    """
    
    # Deploy the model if accuracy is high enough
    deployment_task = deploy_model(
        model=training_task.outputs["model_output"],
        evaluation_metrics=evaluation_task.outputs["evaluation_output"],
        accuracy_threshold=accuracy_threshold,
        kserve_yaml=kserve_yaml
    )

# Compile the pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=bank_marketing_pipeline,
        package_path="bank_marketing_pipeline.yaml"
    )
    
    print("Pipeline compiled successfully. You can now upload 'bank_marketing_pipeline.yaml' to Kubeflow Pipelines.")