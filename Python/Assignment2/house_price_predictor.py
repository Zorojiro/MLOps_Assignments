import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_order = None  # To ensure consistent feature alignment
        self._train_model()

    def _train_model(self):
        # Load dataset (replace with actual dataset path if needed)
        url = "Housing.csv"  # Make sure this file is in the correct path
        df = pd.read_csv(url)

        # Preprocess data: Convert categorical columns to numerical using one-hot encoding
        df = pd.get_dummies(df, drop_first=True)

        # Features and target
        self.feature_order = df.drop(columns=['price']).columns  # Save feature order for later use
        X = df.drop(columns=['price'])  # Drop the target variable from features
        y = df['price']  # Target variable is price

        # Split the data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Train the linear regression model
        self.model.fit(X_train, y_train)

        # Save the model, scaler, and feature order for later use
        joblib.dump(self.model, 'house_price_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.feature_order, 'feature_order.pkl')

    def predict(self, features_dict):
        # Load the saved model, scaler, and feature order
        self.model = joblib.load('house_price_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.feature_order = joblib.load('feature_order.pkl')

        # Convert input dictionary to a DataFrame and align it with the training feature order
        features_df = pd.DataFrame([features_dict])  # Convert dict to DataFrame
        features_df = pd.get_dummies(features_df)  # Apply one-hot encoding
        features_df = features_df.reindex(columns=self.feature_order, fill_value=0)  # Align with training order

        # Scale the features and make the prediction
        features_scaled = self.scaler.transform(features_df)
        prediction = self.model.predict(features_scaled)
        return prediction[0]

predictor = HousePricePredictor()

# Input features as a dictionary
input_features = {
    "area": 2500,
    "bedrooms": 4,
    "bathrooms": 3,
    "stories": 2,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "yes",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 1,
    "prefarea": "yes",
    "furnishingstatus": "furnished"
}

# Make prediction
predicted_price = predictor.predict(input_features)
print(f"Predicted Price: {predicted_price}")
