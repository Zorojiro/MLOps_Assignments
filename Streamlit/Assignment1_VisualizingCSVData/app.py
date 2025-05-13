import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("CSV Data Visualizer")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Read CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Select chart type
    chart_type = st.selectbox("Select chart type", ["Line Chart", "Bar Chart", "Histogram"])

    # Select column(s)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    st.write(f"Available numeric columns: {list(numeric_cols)}")

    if chart_type == "Line Chart":
        col = st.selectbox("Select column for Line Chart", numeric_cols)
        st.line_chart(df[col])

    elif chart_type == "Bar Chart":
        col = st.selectbox("Select column for Bar Chart", numeric_cols)
        st.bar_chart(df[col])

    elif chart_type == "Histogram":
        col = st.selectbox("Select column for Histogram", numeric_cols)
        bins = st.slider("Number of bins", 5, 50, 10)
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=bins, kde=True, ax=ax)
        st.pyplot(fig)
