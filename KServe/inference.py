from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields
from joblib import load

app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="Disease Risk Prediction API",
    description="A simple API to predict disease risk for crops",
)

ns = api.namespace("predict", description="Prediction operations")

# Define the input model for Swagger documentation
prediction_model = api.model(
    "Prediction",
    {
        "crop_name": fields.String(required=True, description="The crop name"),
        "temperature": fields.Float(required=True, description="The temperature"),
        "humidity": fields.Float(required=True, description="The humidity"),
        "soil_moisture": fields.Float(required=True, description="The soil moisture"),
    },
)

# Load the model and encoders
model = load("model.joblib")
crop_label_encoder = load("crop_label_encoder.joblib")
risk_label_encoder = load("risk_label_encoder.joblib")
scaler = load("scaler.joblib")


def predict_disease_risk(crop_name, temperature, humidity, soil_moisture):
    try:
        # Encode the crop name
        crop_name_encoded = crop_label_encoder.transform([crop_name])[0]

        # Prepare the feature vector
        features = scaler.transform(
            [[crop_name_encoded, temperature, humidity, soil_moisture]]
        )

        # Predict the disease risk
        risk_encoded = model.predict(features)[0]

        # Decode the risk
        risk = risk_label_encoder.inverse_transform([risk_encoded])[0]

        return risk
    except Exception as e:
        return str(e)


@ns.route("/")
class Predict(Resource):
    @ns.expect(prediction_model)
    def post(self):
        data = request.json
        crop_name = data["crop_name"]
        temperature = data["temperature"]
        humidity = data["humidity"]
        soil_moisture = data["soil_moisture"]

        # Use the predict_disease_risk function
        risk = predict_disease_risk(crop_name, temperature, humidity, soil_moisture)

        return jsonify({"disease_risk": risk})


api.add_namespace(ns)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)
