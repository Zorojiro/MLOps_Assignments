# Iris Dataset Explorer

This project provides tools to explore, filter, and visualize the famous Iris dataset. It consists of two main components:

1. A FastAPI backend (`iris_filter_api.py`) that provides endpoints for data filtering and visualization
2. A Streamlit frontend (`iris_dashboard.py`) that offers an intuitive user interface to interact with the dataset

## Features

- Filter Iris data based on species
- Visualize distributions of flower measurements (sepal length, sepal width, petal length, petal width)
- Calculate and view statistics for each feature grouped by species
- Download filtered data and visualizations
- Interactive data exploration through a user-friendly dashboard

## Setup and Running

### Running the FastAPI Backend

```bash
cd c:\Users\shubh\Desktop\Work\MLOps_Assignments\Streamlit
python -m python.iris_filter_api
```

This will start the API server at http://127.0.0.1:8051/

API documentation is available at http://127.0.0.1:8051/docs

### Running the Streamlit Dashboard

```bash
cd c:\Users\shubh\Desktop\Work\MLOps_Assignments\Streamlit
streamlit run streamlit/iris_dashboard.py
```

This will start the Streamlit dashboard and automatically open it in your default browser.

## API Endpoints

- **GET /**: Root endpoint with API info
- **GET /iris/info**: Get dataset information
- **GET /iris/species**: Get available species
- **GET /iris/features**: Get available features
- **GET /iris/filter**: Filter data by species
- **GET /iris/visualize**: Generate visualizations for filtered data

## Using the Streamlit Dashboard

1. **Control Panel (Sidebar)**
   - Select species to filter the dataset
   - Choose features to visualize
   - Enable/disable statistics display
   - Apply filters to update the dashboard

2. **Overview Tab**
   - View basic dataset information
   - See a sample of the Iris dataset

3. **Filtered Data Tab**
   - Explore the filtered dataset
   - Download the filtered data as CSV

4. **Visualizations Tab**
   - View distribution histograms for selected features
   - Download visualizations as PNG images

5. **Statistics Tab**
   - View detailed statistics for each feature by species
   - See mean value charts for each species

## Technical Implementation

- **IrisDataFilter Class**: Core class responsible for filtering data and generating visualizations
- **Matplotlib**: Used for creating histograms and bar charts
- **FastAPI**: Provides API endpoints for data access
- **Streamlit**: Creates an interactive web interface
- **Session State**: Maintains filter settings between interactions

## Visualizations

All visualizations are saved in the `iris_visualizations` directory as PNG files.

## Data Description

The Iris dataset contains 150 samples of iris flowers, with the following features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

And three species:
- Setosa
- Versicolor
- Virginica

Each species has 50 samples in the dataset.