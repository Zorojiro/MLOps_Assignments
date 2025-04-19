from typing import Optional
import pandas as pd
import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class HousePricePredictor:
    """
    A class to handle data preprocessing and prediction for house prices.
    """

    def __init__(self, model_path = 'house_price_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.features = None
        self.pipeline = None
        self.target = 'SalePrice'

    def load_data(self):
        """
        Load the House Prices dataset.

        Returns:
            pandas.DataFrame: The loaded house prices dataset
        """
        try:
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/AmesHousing.csv"
            data = pd.read_csv(url)
            print(f"Data downloaded successfully from {url}")
            return data
        except Exception:
            print("Creating sample dataset for demo")

            np.random.seed(42)
            n_samples = 1000

                            # Generate synthetic data for key house features
            data = pd.DataFrame({
                    'LotArea': np.random.randint(1000, 20000, n_samples),
                    'OverallQual': np.random.randint(1, 11, n_samples),
                    'OverallCond': np.random.randint(1, 11, n_samples),
                    'YearBuilt': np.random.randint(1900, 2020, n_samples),
                    'TotalBsmtSF': np.random.randint(0, 3000, n_samples),
                    'GrLivArea': np.random.randint(500, 4000, n_samples),
                    'FullBath': np.random.randint(0, 4, n_samples),
                    'BedroomAbvGr': np.random.randint(0, 6, n_samples),
                    'KitchenAbvGr': np.random.randint(0, 3, n_samples),
                    'GarageCars': np.random.randint(0, 5, n_samples),
                    'GarageArea': np.random.randint(0, 1200, n_samples),
                })
                
                # Generate target variable with some correlation to features
            base_price = 150000
            price = base_price + \
                       data['LotArea'] * 5 + \
                       data['OverallQual'] * 15000 + \
                       data['GrLivArea'] * 100 + \
                       data['GarageCars'] * 8000 + \
                       np.random.normal(0, 20000, n_samples)  # Add some noise
                
            data[self.target] = price
            return data

    def preprocess_data(self, data:pd.DataFrame):
        """
        Preprocess the data by handling missing values and encoding categorical features.

        Args:
            data (pandas.DataFrame): The input dataset

        Returns:
            tuple : Processed X and y data for model training
        """

        # Select relevant numerical features
        numerical_features = [
            'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
            'TotalBsmtSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr',
            'KitchenAbvGr', 'GarageCars', 'GarageArea'
        ]
        
        # Filter features that exist in the dataset
        self.features = [feat for feat in numerical_features if feat in data.columns]

        if self.target not in data.columns:
            raise ValueError(f"Target variable '{self.target}' not found in dataset")
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        self.pipeline = ColumnTransformer(
            transformers = [
                ('num', numeric_transformer, self.features)
            ]
        )

        X = data[self.features]
        y = data[self.target]

        return X , y

    def train(self, X, y):
        """
        Train a linear regression model on the preprocessed data.

        Args:
            X (pd.DataFrame) : Feature data
            y (pd.Series) : Target variable

        Returns:
            self : The trained model instance
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)

        self.model = LinearRegression()
        self.model.fit(X_train_processed, y_train)

        y_pred = self.model.predict(X_test_processed)
        mse = mean_squared_error(y_test, y_pred)

        r2 = r2_score(y_test, y_pred)

        print(f"Model trained successfully")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R2 score: {r2:.2f}")

        return self
    
    def save_model(self):
        """
        Save the trained model to disk

        Returns:
            str : Path to the saved model
        """

        if self.model is None:
            raise ValueError("Model not trained yet")

        model_data = {
            'model':self.model,
            'pipeline':self.pipeline,
            'features': self.features
        }    

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {self.model_path}")
        return self
    
    def load_model(self):
        """
        Load a trained model from disk.

        Returns:
            self.model : The HousePricePredict instance with the loaded model
        """

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found : {self.model_path}")
        
        with open(self.model_path) as f:
            model_data = pickle.load(f)

        self.model = model_data.get("model", None)
        self.pipeline = model_data.get("pipeline", None)
        self.features = model_data.get("features", None)

        print(f"Model loeaded from {self.model_path}")
        return self
    
    def predict(self, input_data):
        """
        Make predictions using the trained model.

        Args:
            input_data (dict or DataFrame): Input features for prediction

        Returns:
            float : Predicted house price
        """

        if self.model is None:
            raise ValueError("Model not trained or loaded yet")
        
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        missing_features = set(self.features) - set(input_data.columns)
        if missing_features:
            for feature in missing_features:
                input_data[feature] = 0

        input_fetures = input_data[self.features]

        input_processed = self.pipeline.transform(input_fetures)

        prediction = self.model.predict(input_processed)

        return prediction[0]
    


app = FastAPI()

class HouseFeaturesInput(BaseModel):
    LotArea: Optional[float] = None
    OverallQual: Optional[int] = None
    OverallCond: Optional[int] = None
    YearBuilt: Optional[int] = None
    TotalBsmtSF: Optional[float] = None
    GrLivArea: Optional[float] = None
    FullBath: Optional[int] = None
    BedroomAbvGr: Optional[int] = None
    KitchenAbvGr: Optional[int] = None
    GarageCars: Optional[int] = None
    GarageArea: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "LotArea": 9000,
                "OverallQual": 7,
                "OverallCond": 5,
                "YearBuilt": 2000,
                "TotalBsmtSF": 900,
                "GrLivArea": 1500,
                "FullBath": 2,
                "BedroomAbvGr": 3,
                "KitchenAbvGr": 1,
                "GarageCars": 2,
                "GarageArea": 480
            }
        }

class PredictionOutput(BaseModel):
    predicted_price: float


predictor = None

@app.on_event("startup")
async def startup_event():
    """
    Init and load the model when the application starts.
    """

    global predictor
    predictor = HousePricePredictor()

    try:
        predictor.load_model()
    except:
        print("Training a new model...")
        data = predictor.load_data()
        X, y = predictor.preprocess_data(data)
        predictor.train(X, y)
        predictor.save_model()

@app.get("/")
async def root():
    """
    Root endpoint that provides basic information about the API.
    """

    return {
        "message" : "House Price Prediction API",
        "endpoint": {
            "/predict":"Make house price predictions",
        }
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(features : HouseFeaturesInput):
    """
    Predict the house price based on the provided features.
    """

    global predict

    if predictor is None:
        raise HTTPException(status_code=500, detail="Model Not Init")
    
    features_dict = features.dict()

    try:
        prediction = predictor.predict(features_dict)
        return {"predicted_price": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error : {str(e)}")
    

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("house_pred:app", host="127.0.0.1", port=8000, reload=True)