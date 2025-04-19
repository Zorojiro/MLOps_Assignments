import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from typing import List, Optional
import uvicorn
from sklearn.datasets import load_wine
import numpy as np

# Create FastAPI app
app = FastAPI(title="Wine Data API", description="API for filtering wine data and creating visualizations")

class WineDataFilter:
    def __init__(self):
        # Load the Wine Quality dataset
        # Using sklearn's wine dataset as a substitute for the Wine Quality dataset
        wine_data = load_wine()
        self.data = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
        self.data['quality'] = wine_data.target  # Using target as quality
        
        # Create directory for saving visualizations if it doesn't exist
        self.viz_dir = "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def filter_by_quality(self, min_quality: int = None, max_quality: int = None) -> pd.DataFrame:
        """Filter wine data based on min and max quality values"""
        filtered_data = self.data.copy()
        
        if min_quality is not None:
            filtered_data = filtered_data[filtered_data['quality'] >= min_quality]
        
        if max_quality is not None:
            filtered_data = filtered_data[filtered_data['quality'] <= max_quality]
        
        return filtered_data
    
    def generate_distribution_plot(self, feature: str, min_quality: int = None, max_quality: int = None, filename: str = None) -> str:
        """Generate a distribution plot for a specific feature based on filtered data"""
        filtered_data = self.filter_by_quality(min_quality, max_quality)
        
        if filename is None:
            quality_info = f"q{min_quality}_to_{max_quality}" if min_quality is not None and max_quality is not None else "all"
            filename = f"{feature}_dist_{quality_info}.png"
        
        filepath = os.path.join(self.viz_dir, filename)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_data[feature], kde=True)
        plt.title(f"Distribution of {feature} (Quality: {min_quality or 'min'}-{max_quality or 'max'})")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath

# Create an instance of WineDataFilter
wine_filter = WineDataFilter()

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to Wine Data API. Use /filter endpoint to filter data and /visualize for visualizations."}

@app.get("/filter")
def filter_wines(min_quality: Optional[int] = Query(None, description="Minimum wine quality"),
                max_quality: Optional[int] = Query(None, description="Maximum wine quality")):
    """Filter wine data based on quality"""
    filtered_data = wine_filter.filter_by_quality(min_quality, max_quality)
    result = {
        "filtered_count": len(filtered_data),
        "data": filtered_data.to_dict(orient="records")
    }
    return result

@app.get("/features")
def get_features():
    """Get the list of available features"""
    return {"features": list(wine_filter.data.columns)}

@app.get("/visualize/{feature}")
def visualize_feature(feature: str, 
                      min_quality: Optional[int] = Query(None, description="Minimum wine quality"),
                      max_quality: Optional[int] = Query(None, description="Maximum wine quality")):
    """Generate and return a visualization for a specific feature"""
    if feature not in wine_filter.data.columns:
        return {"error": f"Feature '{feature}' not found. Available features: {list(wine_filter.data.columns)}"}
    
    try:
        image_path = wine_filter.generate_distribution_plot(feature, min_quality, max_quality)
        return {"image_path": image_path, "message": f"Distribution plot for {feature} generated successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/download-visualization/{feature}")
def download_visualization(feature: str,
                           min_quality: Optional[int] = Query(None, description="Minimum wine quality"),
                           max_quality: Optional[int] = Query(None, description="Maximum wine quality")):
    """Download the visualization for a specific feature"""
    if feature not in wine_filter.data.columns:
        return {"error": f"Feature '{feature}' not found. Available features: {list(wine_filter.data.columns)}"}
    
    try:
        image_path = wine_filter.generate_distribution_plot(feature, min_quality, max_quality)
        return FileResponse(path=image_path, filename=os.path.basename(image_path), media_type="image/png")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn when the script is executed directly
    uvicorn.run("wine_data:app", host="127.0.0.1", port=8000, reload=True)