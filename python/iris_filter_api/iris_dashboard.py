import streamlit as st
import pandas as pd
import requests
import json
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Add the python directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

# Import the Iris data filter functions directly
from python.iris_filter_api.iris_filter_api import load_iris_data, IrisDataFilter

# Set page configuration
st.set_page_config(
    page_title="Iris Dataset Explorer",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state to store data and filter settings
if 'iris_data' not in st.session_state:
    st.session_state.iris_data = load_iris_data()
    st.session_state.iris_filter = IrisDataFilter(st.session_state.iris_data)
    st.session_state.filtered_data = st.session_state.iris_data
    st.session_state.selected_species = st.session_state.iris_filter.species
    st.session_state.selected_features = st.session_state.iris_filter.features
    st.session_state.show_stats = False

# Visualization directory
VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'iris_visualizations')
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Page header
st.title("🌸 Iris Dataset Explorer")
st.markdown("""
This dashboard allows you to explore the famous Iris dataset, filter by species,
visualize feature distributions, and analyze feature statistics.
""")

# Sidebar for filters and controls
st.sidebar.header("Controls")

# Species filter
species_options = st.session_state.iris_filter.species
selected_species = st.sidebar.multiselect(
    "Filter by Species:",
    options=species_options,
    default=species_options
)

# Feature selection
feature_options = st.session_state.iris_filter.features
selected_features = st.sidebar.multiselect(
    "Select Features to Visualize:",
    options=feature_options,
    default=feature_options
)

# Option to show statistics
show_stats = st.sidebar.checkbox("Show Feature Statistics", value=False)

# Apply filters button
if st.sidebar.button("Apply Filters"):
    with st.spinner("Filtering data and generating visualizations..."):
        try:
            # Update session state
            st.session_state.selected_species = selected_species if selected_species else species_options
            st.session_state.selected_features = selected_features if selected_features else feature_options
            st.session_state.show_stats = show_stats
            
            # Filter the data
            st.session_state.filtered_data = st.session_state.iris_filter.filter_by_species(st.session_state.selected_species)
            
            # Success message
            st.sidebar.success("Filters applied successfully!")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# Create tab layout
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Filtered Data", "Visualizations", "Statistics"])

# Tab 1: Overview
with tab1:
    st.header("Dataset Overview")
    
    # Display dataset information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(st.session_state.iris_data))
    
    with col2:
        st.metric("Species", len(st.session_state.iris_filter.species))
    
    with col3:
        st.metric("Features", len(st.session_state.iris_filter.features))
    
    # Brief dataset description
    st.markdown("""
    ### About the Iris Dataset
    
    The Iris dataset is a classic dataset in machine learning and statistics. It contains measurements
    of 150 iris flowers from three different species: setosa, versicolor, and virginica.
    
    For each flower, the following features are measured (in centimeters):
    - Sepal length
    - Sepal width
    - Petal length
    - Petal width
    
    This dataset is widely used for classification tasks and demonstrating machine learning algorithms.
    """)
    
    # Sample of the original data
    st.subheader("Sample Data")
    st.dataframe(st.session_state.iris_data.head(10), use_container_width=True)

# Tab 2: Filtered Data
with tab2:
    st.header("Filtered Data")
    
    # Show currently applied filters
    st.markdown(f"**Selected Species:** {', '.join(st.session_state.selected_species)}")
    st.markdown(f"**Number of Samples:** {len(st.session_state.filtered_data)}")
    
    # Display the filtered data
    st.dataframe(st.session_state.filtered_data, use_container_width=True)
    
    # Add an option to download the filtered data as CSV
    csv = st.session_state.filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="iris_filtered_data.csv",
        mime="text/csv"
    )

# Tab 3: Visualizations
with tab3:
    st.header("Feature Visualizations")
    
    if not st.session_state.selected_features:
        st.warning("Please select at least one feature to visualize.")
    else:
        # Generate visualizations for selected features
        try:
            visualizations = st.session_state.iris_filter.visualize_features(
                st.session_state.filtered_data,
                st.session_state.selected_features,
                save_path=VISUALIZATION_DIR
            )
            
            # Display visualizations in grid layout
            cols = st.columns(2)  # 2 columns for visualizations
            
            for idx, feature in enumerate(st.session_state.selected_features):
                # Load the image from the saved file
                img_path = os.path.join(VISUALIZATION_DIR, f"{feature.replace(' ', '_')}_distribution.png")
                
                if os.path.exists(img_path):
                    # Display in alternating columns
                    with cols[idx % 2]:
                        st.subheader(f"{feature}")
                        st.image(img_path, use_container_width=True)
                        
                        # Add download button for each visualization
                        with open(img_path, "rb") as img_file:
                            btn_key = f"download_{feature.replace(' ', '_')}"
                            st.download_button(
                                label="Download Image",
                                data=img_file,
                                file_name=f"{feature.replace(' ', '_')}_distribution.png",
                                mime="image/png",
                                key=btn_key
                            )
        except Exception as e:
            st.error(f"Error generating visualizations: {str(e)}")

# Tab 4: Statistics
with tab4:
    st.header("Feature Statistics")
    
    if not st.session_state.show_stats:
        st.info("Enable 'Show Feature Statistics' in the sidebar to view statistics.")
    else:
        # Calculate statistics for the filtered data
        try:
            stats = st.session_state.iris_filter.get_feature_statistics(st.session_state.filtered_data)
            
            # Display statistics for each species
            for species in stats:
                st.subheader(f"Statistics for {species.capitalize()}")
                
                # Convert species stats to DataFrame for better display
                species_stats_df = pd.DataFrame.from_dict(
                    {f: {k: v for k, v in fstats.items()} for f, fstats in stats[species].items()},
                    orient='columns'
                )
                
                # Display the statistics
                st.dataframe(species_stats_df, use_container_width=True)
                
                # Simple bar chart of means for each feature
                if st.checkbox(f"Show Mean Values Chart for {species}", key=f"mean_chart_{species}"):
                    feature_means = {f: fstats['mean'] for f, fstats in stats[species].items()}
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.bar(feature_means.keys(), feature_means.values())
                    ax.set_title(f"Mean Feature Values for {species.capitalize()}")
                    ax.set_ylabel("Value (cm)")
                    ax.set_xticklabels(feature_means.keys(), rotation=45)
                    
                    # Add data labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Error calculating statistics: {str(e)}")

# Footer
st.markdown("---")
st.caption("Iris Dataset Explorer | Created with Streamlit and FastAPI")
st.caption("The visualizations are saved in the 'iris_visualizations' directory.")