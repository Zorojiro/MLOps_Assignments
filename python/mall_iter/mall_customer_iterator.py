import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Iterator, Any, Tuple
import os


class ChunkIterator:
    """
    A custom iterator for processing the Mall Customer Segmentation dataset in chunks.
    This allows for efficient memory usage and incremental analysis of large datasets.
    """
    
    def __init__(self, 
                 filepath: str, 
                 chunk_size: int = 50, 
                 columns_to_use: Optional[List[str]] = None,
                 download_if_missing: bool = True):
        """
        Initialize the ChunkIterator.
        
        Parameters:
        -----------
        filepath : str
            Path to the Mall Customer Segmentation dataset CSV file.                        
        chunk_size : int, default=50
            Number of rows to process in each chunk.
        columns_to_use : list of str, optional
            Specific columns to load from the dataset. If None, all columns are loaded.
        download_if_missing : bool, default=True
            If True and the dataset file doesn't exist, download it from a public source.
        """
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.columns_to_use = columns_to_use
        self.current_chunk_index = 0
        self.total_rows_processed = 0
        self.running_stats = {}
        
        # If file doesn't exist and download_if_missing is True, download the dataset
        if not os.path.exists(filepath) and download_if_missing:
            self._download_dataset()
        
        # Ensure the file exists before proceeding
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
        # Get total number of rows in the dataset (for progress tracking)
        self.total_rows = sum(1 for _ in open(filepath)) - 1  # Subtract 1 for header
        
        # Initialize chunk reader
        self._init_chunk_reader()
    
    def _download_dataset(self) -> None:
        """
        Download the Mall Customer Segmentation dataset from a public source.
        """
        import requests
        import io
        
        print("Downloading Mall Customer Segmentation dataset...")
        url = "https://raw.githubusercontent.com/StefanieSenger/Mall_Customers/master/Mall_Customers.csv"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            
            # Save the file
            with open(self.filepath, 'wb') as f:
                f.write(response.content)
                
            print(f"Dataset downloaded and saved to {self.filepath}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    def _init_chunk_reader(self) -> None:
        """
        Initialize or reset the chunk reader.
        """
        self.chunk_reader = pd.read_csv(
            self.filepath,
            chunksize=self.chunk_size,
            usecols=self.columns_to_use
        )
    
    def __iter__(self) -> 'ChunkIterator':
        """
        Return the iterator object itself.
        """
        self._init_chunk_reader()
        self.current_chunk_index = 0
        self.total_rows_processed = 0
        return self
    
    def __next__(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get the next chunk of data and its statistics.
        
        Returns:
        --------
        tuple: (chunk_data, chunk_stats)
            chunk_data: pandas DataFrame containing the current chunk
            chunk_stats: dictionary of statistics for the current chunk
        """
        try:
            chunk = next(self.chunk_reader)
            self.current_chunk_index += 1
            self.total_rows_processed += len(chunk)
            
            # Calculate statistics for this chunk
            chunk_stats = self.calculate_chunk_statistics(chunk)
            
            # Update running statistics
            self._update_running_stats(chunk_stats)
            
            return chunk, chunk_stats
        except StopIteration:
            # End of file reached
            raise StopIteration
    
    def calculate_chunk_statistics(self, chunk: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic statistics for the given chunk.
        
        Parameters:
        -----------
        chunk : pandas DataFrame
            The current chunk of data to analyze.
            
        Returns:
        --------
        dict
            Dictionary containing statistical measures for the chunk.
        """
        stats = {
            'chunk_index': self.current_chunk_index,
            'chunk_size': len(chunk),
            'progress_percent': round((self.total_rows_processed / self.total_rows) * 100, 2),
            'numeric_columns': {},
            'categorical_columns': {}
        }
        
        # Process numeric columns
        numeric_columns = chunk.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            stats['numeric_columns'][col] = {
                'mean': chunk[col].mean(),
                'median': chunk[col].median(),
                'std': chunk[col].std(),
                'min': chunk[col].min(),
                'max': chunk[col].max(),
                'missing_values': chunk[col].isna().sum()
            }
        
        # Process categorical columns
        categorical_columns = chunk.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            value_counts = chunk[col].value_counts()
            stats['categorical_columns'][col] = {
                'unique_count': chunk[col].nunique(),
                'top_values': value_counts.head(3).to_dict(),  # Top 3 most common values
                'missing_values': chunk[col].isna().sum()
            }
        
        return stats
    
    def _update_running_stats(self, chunk_stats: Dict[str, Any]) -> None:
        """
        Update running statistics with the current chunk's statistics.
        
        Parameters:
        -----------
        chunk_stats : dict
            Statistics from the current chunk.
        """
        # Initialize running stats if this is the first chunk
        if not self.running_stats:
            self.running_stats = {
                'total_rows_processed': 0,
                'numeric_columns': {},
                'categorical_columns': {}
            }
            
            # Initialize numeric columns
            for col, stats in chunk_stats['numeric_columns'].items():
                self.running_stats['numeric_columns'][col] = {
                    'sum': 0,
                    'sum_squared': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'missing_values': 0
                }
            
            # Initialize categorical columns
            for col in chunk_stats['categorical_columns']:
                self.running_stats['categorical_columns'][col] = {
                    'value_counts': {},
                    'missing_values': 0
                }
        
        # Update total rows processed
        self.running_stats['total_rows_processed'] += chunk_stats['chunk_size']
        
        # Update numeric column statistics
        for col, stats in chunk_stats['numeric_columns'].items():
            col_data = self.running_stats['numeric_columns'][col]
            
            # Update sum for mean calculation
            col_data['sum'] += stats['mean'] * chunk_stats['chunk_size']
            
            # Update sum_squared for variance/std calculation
            col_data['sum_squared'] += (stats['std']**2 + stats['mean']**2) * chunk_stats['chunk_size']
            
            # Update min/max
            col_data['min'] = min(col_data['min'], stats['min'])
            col_data['max'] = max(col_data['max'], stats['max'])
            
            # Update missing values count
            col_data['missing_values'] += stats['missing_values']
        
        # Update categorical column statistics
        for col, stats in chunk_stats['categorical_columns'].items():
            col_data = self.running_stats['categorical_columns'][col]
            
            # Update value counts
            for value, count in stats['top_values'].items():
                if value in col_data['value_counts']:
                    col_data['value_counts'][value] += count
                else:
                    col_data['value_counts'][value] = count
            
            # Update missing values count
            col_data['missing_values'] += stats['missing_values']
    
    def get_running_statistics(self) -> Dict[str, Any]:
        """
        Get cumulative statistics across all processed chunks.
        
        Returns:
        --------
        dict
            Dictionary containing aggregated statistics across all processed chunks.
        """
        if not self.running_stats or self.running_stats['total_rows_processed'] == 0:
            return {"error": "No data has been processed yet"}
        
        # Create a copy to avoid modifying the internal running stats
        stats = {
            'total_rows_processed': self.running_stats['total_rows_processed'],
            'progress_percent': round((self.total_rows_processed / self.total_rows) * 100, 2),
            'numeric_columns': {},
            'categorical_columns': {}
        }
        
        # Calculate aggregate statistics for numeric columns
        for col, data in self.running_stats['numeric_columns'].items():
            n = self.running_stats['total_rows_processed']
            mean = data['sum'] / n if n > 0 else 0
            
            # Calculate standard deviation
            variance = (data['sum_squared'] / n - mean**2) if n > 0 else 0
            std = np.sqrt(max(0, variance))  # Ensure non-negative
            
            stats['numeric_columns'][col] = {
                'mean': mean,
                'std': std,
                'min': data['min'],
                'max': data['max'],
                'missing_values': data['missing_values'],
                'missing_percent': round((data['missing_values'] / n) * 100, 2) if n > 0 else 0
            }
        
        # Calculate aggregate statistics for categorical columns
        for col, data in self.running_stats['categorical_columns'].items():
            sorted_counts = sorted(data['value_counts'].items(), key=lambda x: x[1], reverse=True)
            
            stats['categorical_columns'][col] = {
                'unique_count': len(data['value_counts']),
                'top_values': dict(sorted_counts[:5]),  # Top 5 most common values
                'missing_values': data['missing_values'],
                'missing_percent': round((data['missing_values'] / stats['total_rows_processed']) * 100, 2)
            }
        
        return stats

    def reset(self) -> None:
        """
        Reset the iterator to the beginning of the dataset.
        """
        self._init_chunk_reader()
        self.current_chunk_index = 0
        self.total_rows_processed = 0
        self.running_stats = {}

    def get_progress(self) -> Dict[str, Any]:
        """
        Get the current progress information.
        
        Returns:
        --------
        dict
            Dictionary containing progress information.
        """
        return {
            'current_chunk': self.current_chunk_index,
            'total_rows_processed': self.total_rows_processed,
            'total_rows': self.total_rows,
            'progress_percent': round((self.total_rows_processed / self.total_rows) * 100, 2) if self.total_rows > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    # Define the path where to save or load the dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "Mall_Customers.csv")
    
    # Create an iterator with chunk size of 100
    iterator = ChunkIterator(dataset_path, chunk_size=100, download_if_missing=True)
    
    # Iterate through chunks and print statistics
    for i, (chunk, stats) in enumerate(iterator):
        print(f"\nProcessing chunk {i+1}:")
        print(f"Rows in chunk: {stats['chunk_size']}")
        print(f"Progress: {stats['progress_percent']}%")
        
        # Print first few rows of each chunk
        print("\nSample data:")
        print(chunk.head(3))
        
        # Print some statistics
        print("\nChunk statistics:")
        for col in stats['numeric_columns']:
            print(f"{col}: Mean = {stats['numeric_columns'][col]['mean']:.2f}, "
                 f"StdDev = {stats['numeric_columns'][col]['std']:.2f}")
        
        # Optional: break after a few chunks for demonstration
        if i >= 2:
            break
    
    # Get overall statistics
    overall_stats = iterator.get_running_statistics()
    print("\n\n===== OVERALL STATISTICS =====")
    print(f"Total rows processed: {overall_stats['total_rows_processed']}")

    print("\nNumeric columns summary:")
    for col, stats in overall_stats['numeric_columns'].items():
        print(f"{col}: Mean = {stats['mean']:.2f}, "
             f"StdDev = {stats['std']:.2f}, "
             f"Range = [{stats['min']}, {stats['max']}]")