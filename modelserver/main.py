import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    """Load and prepare data for training"""
    # Load data
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path, delimiter=';')
    
    # Display basic info
    print(f"Data loaded with shape: {data.shape}")
    print(data.head())
    
    return data

def preprocess_data(data):
    """Preprocess data for model training"""
    # Split features and target
    X = data.drop('y', axis=1)
    y = data['y'].map({'yes': 1, 'no': 0})
    
    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipelines for both numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_model(X_train, y_train, preprocessor):
    """Train a model on the preprocessed data"""
    # Create the model pipeline with preprocessing and classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred

def save_model(model, file_path="./bank_model.joblib"):
    """Save the model to a file"""
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def main():
    # File path
    file_path = './data/bank-full.csv'
    
    # Load data
    data = load_data(file_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, "./bank_model.joblib")
    
    print("Model training complete!")

if __name__ == "__main__":
    main()