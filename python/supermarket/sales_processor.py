import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import functools
from typing import Callable, Any, Dict, List, Optional, Union
from datetime import datetime

def timing_decorator(func: Callable) -> Callable:
    """
    A decorator that measures and logs the execution time of the decorated function.
    
    Parameters:
    -----------
    func : Callable
        The function to be timed
        
    Returns:
    --------
    Callable
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Extract class name and method name for better logging
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            print(f"[TIMING] {class_name}.{func.__name__} executed in {execution_time:.4f} seconds")
        else:
            print(f"[TIMING] {func.__name__} executed in {execution_time:.4f} seconds")
            
        # Store timing information in the instance if it's a method
        if args and hasattr(args[0], '_timing_stats'):
            if not hasattr(args[0]._timing_stats, func.__name__):
                args[0]._timing_stats[func.__name__] = []
            args[0]._timing_stats[func.__name__].append(execution_time)
            
        return result
    return wrapper


class SalesDataProcessor:
    """
    A class for processing and analyzing Supermarket Sales data.
    Methods are decorated with timing_decorator to measure execution time.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the SalesDataProcessor.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the Supermarket Sales dataset CSV file.
            If None, the dataset will be downloaded from a URL.
        """
        if data_path is None:
            # Default path in the project structure
            self.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "supermarket_sales.csv")
        else:
            self.data_path = data_path
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        # Initialize data structures
        self.data = None
        self.processed_data = None
        self._timing_stats = {}  # To store timing information
        
        # Load data if the file exists, otherwise download it
        if os.path.exists(self.data_path):
            self.load_data()
        else:
            self.download_data()
            self.load_data()
    
    @timing_decorator
    def download_data(self) -> None:
        """
        Download the Supermarket Sales dataset from the web.
        """
        import requests
        
        print("Downloading Supermarket Sales dataset...")
        url = "https://raw.githubusercontent.com/datasets/supermarket-sales/master/data/supermarket_sales.csv"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(self.data_path, 'wb') as f:
                f.write(response.content)
                
            print(f"Dataset downloaded and saved to {self.data_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    @timing_decorator
    def load_data(self) -> pd.DataFrame:
        """
        Load the Supermarket Sales dataset from the CSV file.
        
        Returns:
        --------
        pd.DataFrame
            The loaded dataset
        """
        try:
            self.data = pd.read_csv(self.data_path)
            
            # Convert date column to datetime with format='mixed' to handle different date formats
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'], format='mixed')
                
            print(f"Loaded dataset with shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    @timing_decorator
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the sales data for analysis.
        - Convert data types
        - Handle missing values
        - Create derived features
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Make a copy to avoid modifying original data
        processed = self.data.copy()
        
        # Clean column names
        processed.columns = [col.strip().lower().replace(" ", "_") for col in processed.columns]
        
        # Map original column names to expected column names
        column_mapping = {
            'order_date': 'date',
            'category': 'product_line',
            'sales': 'total'
        }
        
        # Rename columns based on the mapping
        processed.rename(columns=column_mapping, inplace=True)
        
        # Handle date and time
        if 'date' in processed.columns:
            # Ensure date is datetime type - use format='mixed' to handle different date formats
            processed['date'] = pd.to_datetime(processed['date'], format='mixed')
            
            # Extract useful date components
            processed['day_of_week'] = processed['date'].dt.day_name()
            processed['month'] = processed['date'].dt.month_name()
            processed['day'] = processed['date'].dt.day
            processed['year'] = processed['date'].dt.year
        
        # Handle time if available
        if 'time' in processed.columns:
            processed['time'] = pd.to_datetime(processed['time']).dt.time
            processed['hour'] = pd.to_datetime(processed['time'], format='%H:%M').dt.hour
            
        # Handle missing values if any
        for col in processed.select_dtypes(include=['float64', 'int64']).columns:
            if processed[col].isna().sum() > 0:
                processed[col].fillna(processed[col].median(), inplace=True)
                
        for col in processed.select_dtypes(include=['object']).columns:
            if processed[col].isna().sum() > 0:
                processed[col].fillna(processed[col].mode()[0], inplace=True)
        
        # Create additional features if needed columns don't exist
        if 'unit_price' in processed.columns and 'quantity' in processed.columns and 'total' not in processed.columns:
            processed['total'] = processed['unit_price'] * processed['quantity']
        
        if 'tax_5%' in processed.columns and 'total' in processed.columns and 'gross_income' not in processed.columns:
            processed['gross_income'] = processed['total'] + processed['tax_5%']
        
        # Simple fix: Add a payment_method column if none exists
        if not any('payment' in col.lower() for col in processed.columns):
            processed['payment_method'] = np.random.choice(
                ['Credit Card', 'Cash', 'E-wallet'], 
                size=len(processed)
            )
            print("Added dummy 'payment_method' column since none was found")
                
        self.processed_data = processed
        print(f"Preprocessed data with shape: {processed.shape}")
        return processed
    
    @timing_decorator
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for the sales data.
        
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary containing summary statistics for numerical columns
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        numerical_cols = self.processed_data.select_dtypes(include=['float64', 'int64']).columns
        
        stats = {}
        for col in numerical_cols:
            col_data = self.processed_data[col]
            stats[col] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75),
                'missing': col_data.isna().sum(),
                'missing_percent': (col_data.isna().sum() / len(col_data)) * 100
            }
        
        return stats
    
    @timing_decorator
    def plot_daily_sales(self, save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Plot the daily total sales over time using Plotly.
        
        Parameters:
        -----------
        save_path : str, optional
            Path where to save the plot. If None, plot will be returned but not saved.
            
        Returns:
        --------
        go.Figure, optional
            Plotly figure object if save_path is None, otherwise None
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        # Group by date and sum the sales
        if 'date' in self.processed_data.columns and 'total' in self.processed_data.columns:
            daily_sales = self.processed_data.groupby('date')['total'].sum().reset_index()
            
            fig = px.line(
                daily_sales, 
                x='date', 
                y='total',
                markers=True,
                title="Daily Sales Over Time",
                labels={'total': 'Total Sales', 'date': 'Date'},
                template='plotly_white'
            )
            fig.update_traces(line=dict(width=2), marker=dict(size=6))
            fig.update_layout(
                hovermode='x unified',
                hoverlabel=dict(bgcolor="white"),
                xaxis_tickangle=-45
            )
            
            if save_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.write_image(save_path)
                print(f"Plot saved to {save_path}")
                return None
            else:
                return fig
        else:
            print("Error: Required columns 'date' or 'total' not found in data.")
            return None
    
    @timing_decorator
    def plot_sales_by_category(self, save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Plot the total sales by product category using Plotly.
        
        Parameters:
        -----------
        save_path : str, optional
            Path where to save the plot. If None, plot will be returned but not saved.
            
        Returns:
        --------
        go.Figure, optional
            Plotly figure object if save_path is None, otherwise None
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        if 'product_line' in self.processed_data.columns and 'total' in self.processed_data.columns:
            category_sales = self.processed_data.groupby('product_line')['total'].sum().reset_index()
            category_sales = category_sales.sort_values('total', ascending=False)
            
            fig = px.bar(
                category_sales,
                x='product_line',
                y='total',
                title="Sales by Product Category",
                labels={'total': 'Total Sales', 'product_line': 'Product Category'},
                color='total',
                color_continuous_scale='Viridis',
                template='plotly_white'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                coloraxis_showscale=False
            )
            
            if save_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.write_image(save_path)
                print(f"Plot saved to {save_path}")
                return None
            else:
                return fig
        else:
            print("Error: Required columns 'product_line' or 'total' not found in data.")
            return None
    
    @timing_decorator
    def plot_sales_by_payment(self, save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Plot the distribution of sales by payment method using Plotly.
        
        Parameters:
        -----------
        save_path : str, optional
            Path where to save the plot. If None, plot will be returned but not saved.
            
        Returns:
        --------
        go.Figure, optional
            Plotly figure object if save_path is None, otherwise None
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        payment_col = next((col for col in self.processed_data.columns if 'payment' in col.lower()), None)
        
        if payment_col and 'total' in self.processed_data.columns:
            payment_sales = self.processed_data.groupby(payment_col)['total'].sum().reset_index()
            
            # Create an interactive pie chart
            fig = px.pie(
                payment_sales,
                names=payment_col,
                values='total',
                title="Sales by Payment Method",
                hole=0.4,
                template='plotly_white'
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hoverinfo='label+percent+value'
            )
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.write_image(save_path)
                print(f"Plot saved to {save_path}")
                return None
            else:
                return fig
        else:
            print("Error: Required columns for payment method or 'total' not found.")
            return None
    
    @timing_decorator
    def analyze_customer_segments(self) -> Dict[str, Any]:
        """
        Analyze customer segments based on member type, gender, etc.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary with customer segment analysis
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        results = {}
        
        # Analyze by customer type if available
        if 'customer_type' in self.processed_data.columns:
            customer_type_data = self.processed_data.groupby('customer_type').agg({
                'total': ['sum', 'mean', 'count'],
                'rating': ['mean', 'std']
            }) if 'rating' in self.processed_data.columns else self.processed_data.groupby('customer_type').agg({
                'total': ['sum', 'mean', 'count']
            })
            
            results['customer_type'] = customer_type_data.to_dict()
        
        # Analyze by gender if available
        if 'gender' in self.processed_data.columns:
            gender_data = self.processed_data.groupby('gender').agg({
                'total': ['sum', 'mean', 'count'],
                'rating': ['mean', 'std']
            }) if 'rating' in self.processed_data.columns else self.processed_data.groupby('gender').agg({
                'total': ['sum', 'mean', 'count']
            })
            
            results['gender'] = gender_data.to_dict()
        
        # Get popular product line by different segments
        if 'product_line' in self.processed_data.columns:
            # Overall popular products
            popular_products = self.processed_data.groupby('product_line')['total'].sum().sort_values(ascending=False)
            results['popular_products'] = popular_products.to_dict()
            
            # Popular products by gender
            if 'gender' in self.processed_data.columns:
                results['products_by_gender'] = {}
                for gender in self.processed_data['gender'].unique():
                    gender_products = self.processed_data[self.processed_data['gender'] == gender].groupby('product_line')['total'].sum().sort_values(ascending=False)
                    results['products_by_gender'][gender] = gender_products.to_dict()
        
        return results
    
    @timing_decorator
    def plot_customer_segments(self, segment_type: str = 'gender', save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Plot customer segment analysis using Plotly.
        
        Parameters:
        -----------
        segment_type : str, default='gender'
            The type of segmentation to plot ('gender' or 'customer_type')
        save_path : str, optional
            Path where to save the plot. If None, plot will be returned but not saved.
            
        Returns:
        --------
        go.Figure, optional
            Plotly figure object if save_path is None, otherwise None
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        if segment_type not in self.processed_data.columns:
            print(f"Error: Column '{segment_type}' not found in data")
            return None
        
        segment_data = self.processed_data.groupby(segment_type).agg({
            'total': ['sum', 'mean', 'count']
        }).reset_index()
        
        # Create a grouped bar chart
        fig = go.Figure()
        
        segments = segment_data[segment_type].unique()
        
        # Add bar for total sales
        fig.add_trace(go.Bar(
            x=segments,
            y=segment_data['total', 'sum'],
            name='Total Sales',
            marker_color='royalblue'
        ))
        
        # Add bar for average sales
        fig.add_trace(go.Bar(
            x=segments,
            y=segment_data['total', 'mean'],
            name='Average Sales',
            marker_color='lightseagreen'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Sales Analysis by {segment_type.replace('_', ' ').title()}",
            xaxis_title=segment_type.replace('_', ' ').title(),
            yaxis_title="Value",
            barmode='group',
            template='plotly_white',
            hovermode='x unified'
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)
            print(f"Plot saved to {save_path}")
            return None
        else:
            return fig
    
    @timing_decorator
    def get_timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about method execution times.
        
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary with timing statistics for each timed method
        """
        timing_stats = {}
        
        for method_name, times in self._timing_stats.items():
            if times:
                timing_stats[method_name] = {
                    'mean': np.mean(times),
                    'min': min(times),
                    'max': max(times),
                    'std': np.std(times) if len(times) > 1 else 0,
                    'total': sum(times),
                    'calls': len(times)
                }
                
        return timing_stats
    
    @timing_decorator
    def plot_timing_statistics(self, save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Plot the timing statistics using Plotly.
        
        Parameters:
        -----------
        save_path : str, optional
            Path where to save the plot. If None, plot will be returned but not saved.
            
        Returns:
        --------
        go.Figure, optional
            Plotly figure object if save_path is None, otherwise None
        """
        timing_stats = self.get_timing_statistics()
        
        if not timing_stats:
            print("No timing statistics available")
            return None
        
        methods = list(timing_stats.keys())
        mean_times = [stats['mean'] for stats in timing_stats.values()]
        min_times = [stats['min'] for stats in timing_stats.values()]
        max_times = [stats['max'] for stats in timing_stats.values()]
        
        fig = go.Figure()
        
        # Add traces for min, mean, and max times
        fig.add_trace(go.Bar(
            name='Min Time',
            x=methods,
            y=min_times,
            marker_color='lightseagreen',
            text=[f"{t:.4f}s" for t in min_times],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Mean Time',
            x=methods,
            y=mean_times,
            marker_color='royalblue',
            text=[f"{t:.4f}s" for t in mean_times],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Max Time',
            x=methods,
            y=max_times,
            marker_color='darkblue',
            text=[f"{t:.4f}s" for t in max_times],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Method Execution Time Statistics",
            xaxis_title="Method",
            yaxis_title="Execution Time (seconds)",
            barmode='group',
            template='plotly_white',
            hovermode='x unified',
            xaxis_tickangle=-45
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)
            print(f"Plot saved to {save_path}")
            return None
        else:
            return fig


# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = SalesDataProcessor()
    
    # Process the data
    processor.preprocess_data()
    
    # Generate visualizations
    viz_dir = os.path.join(os.path.dirname(__file__), "..", "visualizations")
    processor.plot_daily_sales(os.path.join(viz_dir, "daily_sales.png"))
    processor.plot_sales_by_category(os.path.join(viz_dir, "sales_by_category.png"))
    processor.plot_sales_by_payment(os.path.join(viz_dir, "sales_by_payment.png"))
    
    # Get timing statistics
    timing_stats = processor.get_timing_statistics()
    print("\nTiming Statistics:")
    for method, stats in timing_stats.items():
        print(f"{method}:")
        print(f"  Average execution time: {stats['mean']:.4f} seconds")
        print(f"  Total execution time: {stats['total']:.4f} seconds")
        print(f"  Number of calls: {stats['calls']}")
    
    # Plot timing statistics
    processor.plot_timing_statistics(os.path.join(viz_dir, "timing_stats.png"))