import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import os
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(__file__))

# Import our custom ChunkIterator
from mall_customer_iterator import ChunkIterator

class MallCustomerSegmentation:
    """
    Performs customer segmentation analysis on the Mall Customer dataset
    using our custom chunk iterator for incremental processing.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the MallCustomerSegmentation class.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the Mall Customer dataset. If None, a default location will be used.
        """
        if data_path is None:
            self.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "Mall_Customers.csv")
        else:
            self.data_path = data_path
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        # Track data state
        self.data_loaded = False
        self.full_data = None
        self.processed_data = None
        self.scaled_features = None
        self.cluster_labels = None
    
    def load_data_in_chunks(self, chunk_size: int = 50, show_progress: bool = True) -> pd.DataFrame:
        """
        Load and process the mall customer data in chunks using our custom iterator.
        
        Parameters:
        -----------
        chunk_size : int, default=50
            Size of each data chunk.
        show_progress : bool, default=True
            Whether to print progress information.
            
        Returns:
        --------
        pd.DataFrame
            The complete processed dataset.
        """
        # Create a chunk iterator instance
        iterator = ChunkIterator(self.data_path, chunk_size=chunk_size, download_if_missing=True)
        
        # List to collect all processed chunks
        processed_chunks = []
        
        # Process the data chunk by chunk
        start_time = time.time()
        for i, (chunk, stats) in enumerate(iterator):
            # Apply preprocessing to the chunk
            processed_chunk = self._preprocess_chunk(chunk)
            processed_chunks.append(processed_chunk)
            
            # Show progress
            if show_progress and (i % 2 == 0 or i == 0):  # Show every 2 chunks
                progress = iterator.get_progress()
                print(f"Processing chunk {progress['current_chunk']}: "
                      f"{progress['progress_percent']:.1f}% complete "
                      f"({progress['total_rows_processed']}/{progress['total_rows']} rows)")
        
        # Concatenate all processed chunks
        self.full_data = pd.concat(processed_chunks, ignore_index=True)
        self.data_loaded = True
        
        # Get final statistics
        final_stats = iterator.get_running_statistics()
        processing_time = time.time() - start_time
        
        print(f"\nData loading and processing complete. "
              f"Total: {final_stats['total_rows_processed']} rows. "
              f"Time: {processing_time:.2f} seconds.")
        
        return self.full_data
    
    def _preprocess_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a chunk of data. Override this method for custom preprocessing.
        
        This implementation:
        - Cleans column names
        - Checks for missing values
        - Performs basic data validation
        
        Parameters:
        -----------
        chunk : pd.DataFrame
            A chunk of data to process.
            
        Returns:
        --------
        pd.DataFrame
            The processed chunk.
        """
        # Make a copy to avoid modifying original data
        processed = chunk.copy()
        
        # Clean up column names - consistent case and format
        processed.columns = [col.strip().lower().replace(" ", "_") for col in processed.columns]
        
        # Handle missing values if any
        if processed.isnull().sum().sum() > 0:
            # For numeric columns, fill with median
            num_cols = processed.select_dtypes(include=['int64', 'float64']).columns
            for col in num_cols:
                processed[col].fillna(processed[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            cat_cols = processed.select_dtypes(include=['object']).columns
            for col in cat_cols:
                processed[col].fillna(processed[col].mode()[0], inplace=True)
        
        # Convert gender to numerical if it exists
        if 'gender' in processed.columns:
            processed['gender'] = processed['gender'].map({'Male': 0, 'Female': 1})
        
        return processed
    
    def perform_segmentation(self, n_clusters: int = 4, features: List[str] = None) -> Dict[str, Any]:
        """
        Perform customer segmentation using KMeans clustering.
        
        Parameters:
        -----------
        n_clusters : int, default=4
            Number of clusters to create.
        features : list of str, optional
            Features to use for clustering. If None, all numerical features will be used.
            
        Returns:
        --------
        dict
            Dictionary containing segmentation results.
        """
        if not self.data_loaded:
            print("Data not loaded. Loading data first...")
            self.load_data_in_chunks()
        
        # Select features for clustering
        if features is None:
            # Use all numeric features
            features = self.full_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Remove id/customer_id if present
            id_columns = [col for col in features if ('id' in col.lower() and 'age' not in col.lower())]
            for col in id_columns:
                features.remove(col)
        
        print(f"Performing segmentation using {len(features)} features: {', '.join(features)}")
        
        # Extract features for clustering
        X = self.full_data[features].values
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaled_features = X_scaled
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to the dataframe
        self.full_data['cluster'] = self.cluster_labels
        
        # Compute cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_data = self.full_data[self.full_data['cluster'] == i]
            stats = {}
            
            # Calculate statistics for each feature
            for feature in features:
                stats[feature] = {
                    'mean': cluster_data[feature].mean(),
                    'median': cluster_data[feature].median(),
                    'std': cluster_data[feature].std(),
                    'min': cluster_data[feature].min(),
                    'max': cluster_data[feature].max()
                }
            
            cluster_stats[f'Cluster {i}'] = {
                'size': len(cluster_data),
                'percent': (len(cluster_data) / len(self.full_data)) * 100,
                'features': stats
            }
        
        # Create a results object
        results = {
            'data': self.full_data,
            'features_used': features,
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,  # Sum of squared distances to closest centroid
            'cluster_statistics': cluster_stats
        }
        
        return results
    
    def visualize_segmentation(self, results: Dict[str, Any], feature_x: str = None, feature_y: str = None) -> None:
        """
        Visualize the customer segmentation results.
        
        Parameters:
        -----------
        results : dict
            Results from perform_segmentation method.
        feature_x : str, optional
            Feature for x-axis (if None, first feature is used)
        feature_y : str, optional
            Feature for y-axis (if None, second feature is used)
        """
        if feature_x is None:
            feature_x = results['features_used'][0]
        
        if feature_y is None:
            if len(results['features_used']) > 1:
                feature_y = results['features_used'][1]
            else:
                feature_y = results['features_used'][0]  # Fallback to the same feature
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot each cluster with different colors
        data = results['data']
        for i in range(results['n_clusters']):
            cluster_data = data[data['cluster'] == i]
            plt.scatter(
                cluster_data[feature_x],
                cluster_data[feature_y],
                s=60, edgecolor='white', linewidth=0.5,
                label=f'Cluster {i} ({len(cluster_data)} customers)'
            )
        
        # Plot cluster centers
        plt.scatter(
            results['cluster_centers'][:, results['features_used'].index(feature_x)],
            results['cluster_centers'][:, results['features_used'].index(feature_y)],
            s=150, c='black', marker='X', label='Centroids'
        )
        
        plt.title(f'Customer Segments (Total: {len(data)} customers)', fontsize=14)
        plt.xlabel(feature_x.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(feature_y.replace('_', ' ').title(), fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plot_dir = os.path.join(os.path.dirname(__file__), "..", "visualizations")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'mall_clusters_{feature_x}_vs_{feature_y}.png'))
        plt.close()
        
        # Create a pie chart showing segment sizes
        plt.figure(figsize=(8, 8))
        cluster_sizes = [results['cluster_statistics'][f'Cluster {i}']['percent'] for i in range(results['n_clusters'])]
        plt.pie(
            cluster_sizes,
            labels=[f'Cluster {i}\n{size:.1f}%' for i, size in enumerate(cluster_sizes)],
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            explode=[0.05] * results['n_clusters']
        )
        plt.axis('equal')
        plt.title('Customer Segment Size Distribution', fontsize=14)
        plt.tight_layout()
        
        # Save the pie chart
        plt.savefig(os.path.join(plot_dir, 'mall_segment_distribution.png'))
        plt.close()


# Main execution - demonstrate usage
if __name__ == "__main__":
    # Create an instance of the segmentation class
    mall_segmentation = MallCustomerSegmentation("python\Mall_Customers.csv")
    
    # Load the data using our chunk iterator
    print("Loading and processing data in chunks...")
    mall_segmentation.load_data_in_chunks(chunk_size=50)
    
    # Perform customer segmentation
    print("\nPerforming customer segmentation...")
    segmentation_results = mall_segmentation.perform_segmentation(n_clusters=5)
    
    # Display information about the clusters
    print("\n===== Cluster Information =====")
    for cluster, stats in segmentation_results['cluster_statistics'].items():
        print(f"\n{cluster}: {stats['size']} customers ({stats['percent']:.1f}%)")
        if 'annual_income' in stats['features']:
            print(f"  Avg Income: ${stats['features']['annual_income']['mean']:.2f}")
        if 'age' in stats['features']:
            print(f"  Avg Age: {stats['features']['age']['mean']:.1f} years")
        if 'spending_score' in stats['features']:
            print(f"  Avg Spending Score: {stats['features']['spending_score']['mean']:.1f}/100")
    
    # Visualize the segmentation results
    print("\nCreating visualizations...")
    if 'annual_income' in segmentation_results['features_used'] and 'spending_score' in segmentation_results['features_used']:
        mall_segmentation.visualize_segmentation(
            segmentation_results,
            feature_x='annual_income',
            feature_y='spending_score'
        )
        print("Visualization saved to visualizations folder.")
    else:
        mall_segmentation.visualize_segmentation(segmentation_results)
        print("Visualization saved to visualizations folder.")

