import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time
from datetime import datetime

# Add the parent directory to path to import the SalesDataProcessor
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from python.sales_processor import SalesDataProcessor

# Set page configuration
st.set_page_config(
    page_title="Supermarket Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Page title and description
st.title("🛒 Supermarket Sales Dashboard")
st.markdown("""
This dashboard visualizes the Supermarket Sales dataset and demonstrates the timing decorator 
that measures the execution time of various data processing methods.
""")

# Initialize session state variables if they don't exist
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.data_loaded = False
    st.session_state.data_processed = False

# Sidebar
st.sidebar.header("Controls")

# Button to initialize data processor
if st.sidebar.button("Load Data"):
    with st.spinner("Loading and processing data..."):
        # Initialize the SalesDataProcessor
        st.session_state.processor = SalesDataProcessor()
        st.session_state.data_loaded = True
        st.session_state.data_processed = True
        st.success("Data loaded successfully!")

# Main content - only show if data is loaded
if st.session_state.data_loaded and st.session_state.processor:
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Sales Analysis", "Customer Segments", "Timing Stats"])
    
    processor = st.session_state.processor
    
    with tab1:
        st.header("Dataset Overview")
        
        # Get statistics
        stats = processor.get_summary_statistics()
        
        # Show raw data
        with st.expander("Show Raw Data Sample"):
            st.dataframe(processor.processed_data.head(10))
        
        # Show summary statistics
        st.subheader("Summary Statistics")
        
        # Create 3 columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        # Display key metrics
        if 'total' in stats:
            # Calculate the total sales sum
            total_sales_sum = processor.processed_data['total'].sum()
            col1.metric("Total Sales", f"${total_sales_sum:,.2f}")
        if 'quantity' in stats:
            # Calculate the total quantity sum if it exists
            if 'quantity' in processor.processed_data.columns:
                total_quantity = processor.processed_data['quantity'].sum()
                col2.metric("Total Items Sold", f"{int(total_quantity):,}")
        
        # Number of unique customers if customer_id exists
        if 'customer_id' in processor.processed_data.columns:
            unique_customers = processor.processed_data['customer_id'].nunique()
            col3.metric("Unique Customers", f"{unique_customers:,}")
        
        # Show full statistics
        selected_column = st.selectbox("Select column for detailed statistics:", 
                                      options=list(stats.keys()))
        
        if selected_column:
            st.write(f"Statistics for {selected_column}:")
            stat_df = pd.DataFrame({k: [v] for k, v in stats[selected_column].items()}).T
            stat_df.columns = ["Value"]
            st.dataframe(stat_df)
            
            # Add a histogram for the selected column
            if selected_column in processor.processed_data.columns:
                fig = px.histogram(
                    processor.processed_data, 
                    x=selected_column,
                    title=f"Distribution of {selected_column}",
                    nbins=30,
                    color_discrete_sequence=['#3366CC']
                )
                fig.update_layout(
                    xaxis_title=selected_column.replace('_', ' ').title(),
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Sales Analysis")
        
        # Create interactive plotly visualizations
        st.subheader("Daily Sales Over Time")
        
        if st.button("Generate Daily Sales Plot"):
            with st.spinner("Generating plot..."):
                if 'date' in processor.processed_data.columns and 'total' in processor.processed_data.columns:
                    daily_sales = processor.processed_data.groupby('date')['total'].sum().reset_index()
                    
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
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Required columns 'date' or 'total' not found in data.")
        
        st.subheader("Sales by Product Category")
        
        if st.button("Generate Category Sales Plot"):
            with st.spinner("Generating plot..."):
                if 'product_line' in processor.processed_data.columns and 'total' in processor.processed_data.columns:
                    category_sales = processor.processed_data.groupby('product_line')['total'].sum().reset_index()
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
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Required columns 'product_line' or 'total' not found in data.")
        
        st.subheader("Sales by Payment Method")
        
        if st.button("Generate Payment Method Plot"):
            with st.spinner("Generating plot..."):
                payment_col = next((col for col in processor.processed_data.columns if 'payment' in col.lower()), None)
                
                if payment_col and 'total' in processor.processed_data.columns:
                    payment_sales = processor.processed_data.groupby(payment_col)['total'].sum().reset_index()
                    
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
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Required columns for payment method or 'total' not found.")
    
    with tab3:
        st.header("Customer Segment Analysis")
        
        if st.button("Analyze Customer Segments"):
            with st.spinner("Analyzing customer segments..."):
                segment_results = processor.analyze_customer_segments()
                
                # Display segment information
                if 'customer_type' in segment_results:
                    st.subheader("Analysis by Customer Type")
                    customer_type_df = pd.DataFrame(segment_results['customer_type'])
                    st.dataframe(customer_type_df)
                    
                    # Extract customer type data for visualization
                    if ('total', 'sum') in customer_type_df.columns:
                        customer_types = customer_type_df.index.tolist()
                        sales_values = [customer_type_df.loc[ctype, ('total', 'sum')] for ctype in customer_types]
                        
                        fig = px.pie(
                            names=customer_types,
                            values=sales_values,
                            title="Sales Distribution by Customer Type",
                            template='plotly_white'
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                
                if 'gender' in segment_results:
                    st.subheader("Analysis by Gender")
                    gender_df = pd.DataFrame(segment_results['gender'])
                    st.dataframe(gender_df)
                    
                    # Create gender comparison visualization
                    if ('total', 'sum') in gender_df.columns and ('total', 'count') in gender_df.columns:
                        genders = gender_df.index.tolist()
                        
                        # Create a comparison bar chart
                        fig = go.Figure()
                        
                        # Add bar for total sales
                        fig.add_trace(go.Bar(
                            x=genders,
                            y=[gender_df.loc[gender, ('total', 'sum')] for gender in genders],
                            name='Total Sales',
                            marker_color='#3366CC'
                        ))
                        
                        # Add bar for average sales
                        fig.add_trace(go.Bar(
                            x=genders,
                            y=[gender_df.loc[gender, ('total', 'mean')] for gender in genders],
                            name='Average Sales',
                            marker_color='#33CC36'
                        ))
                        
                        fig.update_layout(
                            title="Sales by Gender",
                            barmode='group',
                            template='plotly_white',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                if 'popular_products' in segment_results:
                    st.subheader("Popular Products")
                    popular_products = pd.Series(segment_results['popular_products'])
                    
                    # Create an interactive bar chart
                    fig = px.bar(
                        x=popular_products.index,
                        y=popular_products.values,
                        title="Popular Products by Sales Value",
                        labels={'x': 'Product Category', 'y': 'Total Sales'},
                        color=popular_products.values,
                        color_continuous_scale='Viridis',
                        template='plotly_white'
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        coloraxis_showscale=False,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Method Execution Timing Statistics")
        
        if st.button("Refresh Timing Stats"):
            with st.spinner("Calculating timing statistics..."):
                timing_stats = processor.get_timing_statistics()
                
                # Display timing statistics in a table
                timing_data = []
                for method, stats in timing_stats.items():
                    timing_data.append({
                        'Method': method,
                        'Mean Time (s)': f"{stats['mean']:.4f}",
                        'Min Time (s)': f"{stats['min']:.4f}",
                        'Max Time (s)': f"{stats['max']:.4f}",
                        'Total Time (s)': f"{stats['total']:.4f}",
                        'Calls': stats['calls']
                    })
                
                timing_df = pd.DataFrame(timing_data)
                st.dataframe(timing_df)
                
                # Create an interactive bar chart for timing stats
                method_names = [d['Method'] for d in timing_data]
                mean_times = [float(d['Mean Time (s)'].replace(',', '')) for d in timing_data]
                
                fig = px.bar(
                    x=method_names,
                    y=mean_times,
                    title="Average Execution Time by Method",
                    labels={'x': 'Method', 'y': 'Time (seconds)'},
                    color=mean_times,
                    color_continuous_scale='Viridis',
                    template='plotly_white'
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    coloraxis_showscale=False,
                    hovermode='x unified'
                )
                # Add data labels on the bars
                fig.update_traces(
                    texttemplate='%{y:.4f}s',
                    textposition='outside'
                )
                st.plotly_chart(fig, use_container_width=True)

        # Add a button to demonstrate multiple method calls
        if st.button("Run Performance Test"):
            with st.spinner("Running multiple method calls to benchmark performance..."):
                # Run multiple methods to generate timing stats
                for _ in range(3):
                    processor.get_summary_statistics()
                    
                for _ in range(2):
                    processor.analyze_customer_segments()
                
                # Display updated timing stats
                timing_stats = processor.get_timing_statistics()
                
                # Display timing statistics in a table
                timing_data = []
                for method, stats in timing_stats.items():
                    timing_data.append({
                        'Method': method,
                        'Mean Time (s)': f"{stats['mean']:.4f}",
                        'Min Time (s)': f"{stats['min']:.4f}",
                        'Max Time (s)': f"{stats['max']:.4f}",
                        'Total Time (s)': f"{stats['total']:.4f}",
                        'Calls': stats['calls']
                    })
                
                timing_df = pd.DataFrame(timing_data)
                st.dataframe(timing_df)
                
                # Create a detailed timing visualization
                fig = go.Figure()
                
                for method, stats in timing_stats.items():
                    fig.add_trace(go.Bar(
                        name=method,
                        x=['Min', 'Mean', 'Max'],
                        y=[stats['min'], stats['mean'], stats['max']],
                        text=[f"{stats['min']:.4f}s", f"{stats['mean']:.4f}s", f"{stats['max']:.4f}s"],
                        textposition='outside'
                    ))
                
                fig.update_layout(
                    title="Method Execution Time Comparison",
                    xaxis_title="Time Metric",
                    yaxis_title="Time (seconds)",
                    barmode='group',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)

else:
    # Show instructions if data is not loaded
    st.info("👈 Please click 'Load Data' in the sidebar to get started.")
    
    # Display some information about the app
    st.markdown("""
    ### About This Dashboard
    
    This application demonstrates the use of a custom timing decorator to measure the execution time of methods in a data processing class.
    
    #### Key Features:
    
    - **Timing Decorator**: Measures and records execution time of class methods
    - **Data Processing**: Loads and processes supermarket sales data
    - **Interactive Visualization**: Creates interactive Plotly charts to analyze sales patterns
    - **Performance Metrics**: Shows performance statistics for all timed methods
    
    Click the 'Load Data' button to start.
    """)

# Footer
st.markdown("---")
st.caption("Supermarket Sales Dashboard | Created with timing decorators and Plotly interactive visualizations")