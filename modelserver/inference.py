import joblib
import pandas as pd
import numpy as np

class BankMarketingPredictor:
    def __init__(self, model_path="./bank_model.joblib"):
        """Initialize the predictor with a trained model"""
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_input(self, input_data):
        """Preprocess input data before prediction
        
        Args:
            input_data: Dictionary or JSON containing input features
        
        Returns:
            DataFrame: Preprocessed data ready for prediction
        """
        # Convert input dictionary to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)
        
        # Define required features based on the specified 16 features
        required_features = [
            'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'poutcome', 'age', 'balance', 'day',
            'duration', 'campaign', 'pdays', 'previous'
        ]
        
        # Check for missing required columns
        missing_cols = set(required_features) - set(input_df.columns)
        if missing_cols:
            print(f"Warning: Missing required columns {missing_cols}. Using default values.")
            
            # Default values for missing columns
            default_values = {
                'job': 'unknown',
                'marital': 'unknown',
                'education': 'unknown',
                'default': 'unknown',
                'housing': 'unknown',
                'loan': 'unknown',
                'contact': 'unknown',
                'month': 'unknown',
                'poutcome': 'unknown',
                'age': 40,
                'balance': 0,
                'day': 15,
                'duration': 180,
                'campaign': 1,
                'pdays': 999,
                'previous': 0
            }
            
            # Add the missing columns with default values
            for col in missing_cols:
                input_df[col] = default_values.get(col, 'unknown')
        
        # Ensure all features are using the expected data types
        numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        for feature in numeric_features:
            if feature in input_df.columns:
                input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').fillna(0)
        
        return input_df
    
    def predict(self, input_data):
        """Make predictions using the loaded model
        
        Args:
            input_data: Dictionary or JSON containing input features
        
        Returns:
            dict: Prediction results with probability
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Preprocess input data
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0]
            
            result = {
                "prediction": "yes" if prediction == 1 else "no",
                "probability": {
                    "no": float(probability[0]),
                    "yes": float(probability[1])
                }
            }
            
            return result
        
        except Exception as e:
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    predictor = BankMarketingPredictor("./bank_model.joblib")
    
    # Example input with all 16 features
    sample_input = {
        "job": "blue-collar",
        "marital": "married", 
        "education": "basic.4y",
        "default": "no", 
        "housing": "yes", 
        "loan": "no",
        "contact": "telephone", 
        "month": "may", 
        "poutcome": "nonexistent",
        "age": 40,
        "balance": 1500,
        "day": 15,
        "duration": 180,
        "campaign": 1,
        "pdays": 999,
        "previous": 0
    }
    
    prediction = predictor.predict(sample_input)
    print(f"Prediction: {prediction}")