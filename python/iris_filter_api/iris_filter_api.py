import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import uvicorn
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.datasets import load_iris

def load_iris_data():
    """
    Load and prepare the Iris dataset
    
    Returns:
        pandas.DataFrame: The prepared Iris dataset
    """
    # Load the Iris dataset from scikit-learn
    iris = load_iris()
    
    # Create a DataFrame from the data
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Add the target column with species names
    df['species'] = [iris.target_names[i] for i in iris.target]
    
    return df

class IrisDataFilter:
    """
    A class to filter the Iris dataset based on species and visualize feature distributions
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the filter with the Iris dataset
        
        Args:
            data (pd.DataFrame): The Iris dataset
        """
        self.data = data
        self.features = [col for col in data.columns if col != 'species']
        self.species = data['species'].unique().tolist()
        
    def filter_by_species(self, selected_species: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter the dataset by selected species
        
        Args:
            selected_species (Optional[List[str]]): List of species to include. If None, all species are included
            
        Returns:
            pd.DataFrame: The filtered dataset
        """
        if selected_species is None or len(selected_species) == 0:
            return self.data
        
        # Validate species names
        invalid_species = [s for s in selected_species if s not in self.species]
        if invalid_species:
            raise ValueError(f"Invalid species: {invalid_species}. Available species: {self.species}")
            
        return self.data[self.data['species'].isin(selected_species)]
    
    def visualize_features(self, 
                          filtered_data: pd.DataFrame, 
                          features: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Create visualizations of feature distributions for the filtered data
        
        Args:
            filtered_data (pd.DataFrame): The filtered dataset
            features (Optional[List[str]]): List of features to visualize. If None, all features are visualized
            save_path (Optional[str]): Directory to save the visualizations. If None, visualizations are not saved to disk
            
        Returns:
            Dict[str, str]: Dictionary with feature names as keys and base64 encoded images as values
        """
        if features is None:
            features = self.features
        
        # Validate feature names
        invalid_features = [f for f in features if f not in self.features]
        if invalid_features:
            raise ValueError(f"Invalid features: {invalid_features}. Available features: {self.features}")
        
        # Create a directory for saved visualizations if needed
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            
        visualizations = {}
        
        for feature in features:
            plt.figure(figsize=(10, 6))
            
            # Group by species and plot histograms
            for species in filtered_data['species'].unique():
                species_data = filtered_data[filtered_data['species'] == species]
                plt.hist(species_data[feature], alpha=0.5, label=species)
            
            plt.title(f'Distribution of {feature} by Species')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            
            # Save to file if a path is provided
            if save_path is not None:
                file_path = os.path.join(save_path, f"{feature.replace(' ', '_')}_distribution.png")
                plt.savefig(file_path)
                visualizations[feature] = file_path
            
            # Also convert to base64 for API responses
            img_buf = BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_data = base64.b64encode(img_buf.read()).decode('utf-8')
            visualizations[feature] = img_data
            
            plt.close()
            
        return visualizations
    
    def get_feature_statistics(self, filtered_data: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get statistics for each feature grouped by species
        
        Args:
            filtered_data (pd.DataFrame): The filtered dataset
            
        Returns:
            Dict: Dictionary with feature statistics by species
        """
        stats = {}
        
        for species in filtered_data['species'].unique():
            species_data = filtered_data[filtered_data['species'] == species]
            species_stats = {}
            
            for feature in self.features:
                feature_stats = {
                    'mean': float(species_data[feature].mean()),
                    'median': float(species_data[feature].median()),
                    'std': float(species_data[feature].std()),
                    'min': float(species_data[feature].min()),
                    'max': float(species_data[feature].max())
                }
                species_stats[feature] = feature_stats
                
            stats[species] = species_stats
            
        return stats

# Create a directory for visualizations
VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'iris_visualizations')
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Load the dataset at startup
iris_data = load_iris_data()
iris_filter = IrisDataFilter(iris_data)

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Iris Dataset API",
        "endpoints": [
            {"path": "/", "description": "This information"},
            {"path": "/iris/info", "description": "Get dataset information"},
            {"path": "/iris/species", "description": "Get available species"},
            {"path": "/iris/features", "description": "Get available features"},
            {"path": "/iris/filter", "description": "Filter data by species"},
            {"path": "/iris/visualize", "description": "Generate visualizations"}
        ]
    }

@app.get("/iris/info")
async def get_dataset_info():
    """Get information about the dataset"""
    return {
        "total_samples": len(iris_data),
        "features": iris_filter.features,
        "species": iris_filter.species,
        "sample_data": iris_data.head(5).to_dict(orient='records')
    }

@app.get("/iris/species")
async def get_species():
    """Get list of available species"""
    return {"species": iris_filter.species}

@app.get("/iris/features")
async def get_features():
    """Get list of available features"""
    return {"features": iris_filter.features}

@app.get("/iris/filter")
async def filter_data(
    species: Optional[List[str]] = Query(None, description="List of species to include")
):
    """
    Filter dataset by selected species
    
    Args:
        species: List of species to include. If not provided, all species are included.
    
    Returns:
        Filtered dataset
    """
    try:
        filtered_data = iris_filter.filter_by_species(species)
        return {
            "filtered_count": len(filtered_data),
            "selected_species": species if species else iris_filter.species,
            "data": filtered_data.to_dict(orient='records')
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/iris/visualize")
async def visualize_data(
    species: Optional[List[str]] = Query(None, description="List of species to include"),
    features: Optional[List[str]] = Query(None, description="List of features to visualize"),
    include_stats: bool = Query(False, description="Include feature statistics")
):
    """
    Generate visualizations for the filtered data
    
    Args:
        species: List of species to include. If not provided, all species are included.
        features: List of features to visualize. If not provided, all features are visualized.
        include_stats: Whether to include feature statistics.
    
    Returns:
        Visualizations and optionally statistics
    """
    try:
        filtered_data = iris_filter.filter_by_species(species)
        
        # Save visualizations to disk
        visualizations = iris_filter.visualize_features(
            filtered_data, 
            features, 
            save_path=VISUALIZATION_DIR
        )
        
        response = {
            "filtered_count": len(filtered_data),
            "selected_species": species if species else iris_filter.species,
            "visualizations": {
                feature: f"data:image/png;base64,{image_data}" 
                for feature, image_data in visualizations.items()
            },
            "visualization_paths": {
                feature: os.path.join(VISUALIZATION_DIR, f"{feature.replace(' ', '_')}_distribution.png")
                for feature in visualizations.keys()
            }
        }
        
        # Include statistics if requested
        if include_stats:
            response["statistics"] = iris_filter.get_feature_statistics(filtered_data)
            
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("iris_filter_api:app", host="127.0.0.1", port=8051, reload=True)