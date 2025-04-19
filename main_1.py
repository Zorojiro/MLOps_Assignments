import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import plotly.express as px


st.title("CSV Explorer")

data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

if data not in st.session_state:
    st.session_state[df] = df

st.sidebar.title("Sidebar")

radio = st.sidebar.radio(
    "Select an option",
    ("Raw Data", "Data Overview", "Plot Scatter Graph", "Plot Pie Chart")
)

if radio == "Raw Data":
    st.write("### Raw Data")
    st.dataframe(st.session_state[df])

elif radio == "Plot Scatter Graph":
    st.write("### Plot Data")
    x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)

    fig = px.scatter(df, x=x_axis, y=y_axis, color='target')
    st.plotly_chart(fig)

elif radio == "Plot Pie Chart":
    st.write("### Plot Pie Chart")
    column = st.sidebar.selectbox("Select Column", df.columns)
    fig = px.pie(df, names=column, values='target')
    st.plotly_chart(fig)

elif radio == "Data Overview":
    st.write("### Data Overview")
    st.write(st.session_state[df].describe())
