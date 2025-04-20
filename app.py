from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import sys
import json
import pandas as pd
import io

# Add the modelserver directory to the path to import the predictor
sys.path.append(os.path.join(os.path.dirname(__file__), 'modelserver'))
from modelserver.inference import BankMarketingPredictor

# Create FastAPI app
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API for predicting term deposit subscriptions",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize the predictor
model_path = os.path.join(os.path.dirname(__file__), 'modelserver', 'bank_model.joblib')
predictor = BankMarketingPredictor(model_path)

# Define data models
class PredictionInput(BaseModel):
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    poutcome: str
    age: int
    balance: int
    day: int
    duration: int
    campaign: int
    pdays: int
    previous: int

    class Config:
        json_schema_extra = {
            "example": {
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
        }

class PredictionResult(BaseModel):
    prediction: str
    probability: Dict[str, float]

class BatchPredictionResult(BaseModel):
    batch_predictions: List[Dict[str, Any]]
    total_processed: int

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResult)
async def predict(data: PredictionInput):
    """API endpoint for making predictions"""
    try:
        # Convert Pydantic model to dictionary
        input_data = data.dict()
        
        # Make prediction
        result = predictor.predict(input_data)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict", response_model=BatchPredictionResult)
async def batch_predict(file: UploadFile = File(...)):
    """API endpoint for batch predictions"""
    try:
        # Check file content type
        if not file.content_type.startswith('text/csv') and not file.content_type == 'application/vnd.ms-excel':
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
            
        # Read CSV file
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode('utf-8')), delimiter=';')
        
        # Process each row and make predictions
        results = []
        for _, row in data.iterrows():
            input_data = row.to_dict()
            prediction = predictor.predict(input_data)
            results.append({
                'input': input_data,
                'prediction': prediction
            })
        
        return {
            'batch_predictions': results,
            'total_processed': len(results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)