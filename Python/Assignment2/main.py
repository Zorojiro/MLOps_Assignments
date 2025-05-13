from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from house_price_predictor import HousePricePredictor
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve static files from 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html page
@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("templates/index.html", "r") as f:  # Assuming index.html is in a 'templates' folder
        return f.read()

# Define the input data model
class HouseFeatures(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

# Initialize the model
predictor = HousePricePredictor()

@app.post("/predict-price/")
async def predict_price(features: HouseFeatures):
    print("Received data:", features)  # Log the received data
    feature_values = [
        features.area, features.bedrooms, features.bathrooms, features.stories,
        features.mainroad == "yes", features.guestroom == "yes", features.basement == "yes",
        features.hotwaterheating == "yes", features.airconditioning == "yes",
        features.parking, features.prefarea == "yes", features.furnishingstatus == "furnished"
    ]
    prediction = predictor.predict(feature_values)
    return {"predicted_price": prediction}
