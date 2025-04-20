import kserve
from typing import Dict
import joblib
import pandas as pd
import numpy as np
import os

class BankMarketingModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False
    
    def load(self):
        model_path = os.path.join(
            kserve.Storage.download(self.name),
            "bank_model.joblib"
        )
        self.model = joblib.load(model_path)
        self.ready = True
    
    def preprocess(self, inputs: Dict) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        # Convert input dictionary to DataFrame
        if isinstance(inputs, dict):
            input_df = pd.DataFrame([inputs])
        else:
            input_df = pd.DataFrame(inputs)
        
        # Ensure all required columns are present
        expected_columns = ['job','marital','education','default',
                            'housing','loan','contact','month',
                            'poutcome','age','balance','day','duration',
                            'campaign','pdays','previous']
        
        # Check for missing columns
        missing_cols = set(expected_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = "unknown"  # Default value
        
        return input_df
    
    def predict(self, inputs: Dict) -> Dict:
        """Make predictions using the loaded model
        
        Args:
            inputs: Dictionary containing input features
        
        Returns:
            dict: Prediction results with probability
        """
        try:
            # Preprocess input data
            processed_data = self.preprocess(inputs)
            
            # Make prediction
            predictions = self.model.predict(processed_data)
            probabilities = self.model.predict_proba(processed_data)
            
            results = []
            for i, prediction in enumerate(predictions):
                result = {
                    "prediction": "yes" if prediction == 1 else "no",
                    "probability": {
                        "no": float(probabilities[i][0]),
                        "yes": float(probabilities[i][1])
                    }
                }
                results.append(result)
            
            return {"predictions": results}
        
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    model = BankMarketingModel("bank-marketing-model")
    kserve.ModelServer().start([model])