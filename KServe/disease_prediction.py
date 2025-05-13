import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create a synthetic dataset
data = {
    "crop_name": [
        "wheat",
        "rice",
        "maize",
        "wheat",
        "rice",
        "maize",
        "wheat",
        "rice",
        "maize",
        "wheat",
        "rice",
        "maize",
    ],
    "temperature": [20, 25, 22, 21, 24, 23, 19, 26, 21, 20, 25, 22],
    "humidity": [30, 50, 45, 32, 48, 47, 31, 52, 44, 30, 50, 45],
    "soil_moisture": [40, 60, 55, 42, 58, 57, 41, 62, 54, 40, 60, 55],
    "disease_risk": [
        "low",
        "high",
        "medium",
        "low",
        "high",
        "medium",
        "low",
        "high",
        "medium",
        "low",
        "high",
        "medium",
    ],
}

df = pd.DataFrame(data)

# Encode categorical variable for crop_name
crop_label_encoder = LabelEncoder()
df["crop_name"] = crop_label_encoder.fit_transform(df["crop_name"])
# Encode target variable
risk_label_encoder = LabelEncoder()
df["disease_risk"] = risk_label_encoder.fit_transform(df["disease_risk"])

# Features and target variable
X = df[["crop_name", "temperature", "humidity", "soil_moisture"]]
y = df["disease_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

dump(model, "model.joblib")
dump(scaler, "scaler.joblib")
dump(crop_label_encoder, "crop_label_encoder.joblib")
dump(risk_label_encoder, "risk_label_encoder.joblib")


# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed classification report
print(classification_report(y_test, y_pred))
