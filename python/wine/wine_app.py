import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# FastAPI server URL
API_URL = "http://127.0.0.1:8000"

# Set page configuration
st.set_page_config(
    page_title="Wine Data Explorer",
    page_icon="🍷",
    layout="wide"
)

# Page title
st.title("🍷 Wine Quality Data Explorer")
st.write("Filter wine data based on quality and visualize feature distributions")

# Sidebar for filters
st.sidebar.header("Filters")

# Get available features from the API (this will work once the API is running)
try:
    features_response = requests.get(f"{API_URL}/features")
    if features_response.status_code == 200:
        available_features = features_response.json()["features"]
    else:
        # Default features if API is not available
        available_features = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", 
                            "magnesium", "total_phenols", "flavanoids", 
                            "nonflavanoid_phenols", "proanthocyanins", 
                            "color_intensity", "hue", "od280/od315_of_diluted_wines", 
                            "proline", "quality"]
except:
    # Default features if API is not available
    available_features = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", 
                        "magnesium", "total_phenols", "flavanoids", 
                        "nonflavanoid_phenols", "proanthocyanins", 
                        "color_intensity", "hue", "od280/od315_of_diluted_wines", 
                        "proline", "quality"]

# Quality filter sliders
min_quality = st.sidebar.slider("Minimum Quality", 0, 9, 0)
max_quality = st.sidebar.slider("Maximum Quality", 0, 9, 9)

# Selected features for visualization
st.sidebar.header("Visualize Features")
selected_feature = st.sidebar.selectbox("Select feature to visualize", available_features)

# Main content
st.header("Filtered Wine Data")

# Button to trigger data fetch
if st.button("Fetch Wine Data"):
    # Call API to get filtered data
    try:
        response = requests.get(f"{API_URL}/filter", params={"min_quality": min_quality, "max_quality": max_quality})
        
        if response.status_code == 200:
            data = response.json()
            st.success(f"Found {data['filtered_count']} wines matching your criteria")
            
            # Display data as a table
            if data['filtered_count'] > 0:
                df = pd.DataFrame(data['data'])
                st.dataframe(df)
            else:
                st.info("No wines match the selected criteria")
        else:
            st.error("Error fetching data from API")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        st.info("Make sure the FastAPI server is running at " + API_URL)

# Visualization section
st.header("Feature Visualization")
st.write(f"Displaying distribution for: **{selected_feature}**")

# Button to generate visualization
if st.button("Generate Visualization"):
    try:
        # Create a spinner while waiting for the API response
        with st.spinner("Generating visualization..."):
            # Call API to generate visualization
            viz_response = requests.get(
                f"{API_URL}/download-visualization/{selected_feature}",
                params={"min_quality": min_quality, "max_quality": max_quality},
                stream=True
            )
            
            if viz_response.status_code == 200:
                # Convert the image data to a format Streamlit can display
                image = Image.open(io.BytesIO(viz_response.content))
                st.image(image, caption=f"Distribution of {selected_feature} (Quality: {min_quality}-{max_quality})")
            else:
                error_message = viz_response.json().get("error", "Unknown error")
                st.error(f"Error generating visualization: {error_message}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        st.info("Make sure the FastAPI server is running at " + API_URL)

# Instructions section
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Use the sliders to filter wines by quality
2. Click 'Fetch Wine Data' to display filtered data
3. Select a feature to visualize its distribution
4. Click 'Generate Visualization' to create and display the plot
""")

# Information about the API
st.sidebar.header("API Information")
st.sidebar.info(f"""
The data is served by a FastAPI endpoint at:
{API_URL}

Available endpoints:
- /filter: Filter wines by quality
- /visualize/{{feature}}: Generate visualization
- /download-visualization/{{feature}}: Download visualization
- /features: List all available features
""")

# Footer
st.markdown("---")
st.caption("Wine Data Explorer | Powered by FastAPI and Streamlit")