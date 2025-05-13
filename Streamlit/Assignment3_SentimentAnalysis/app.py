import streamlit as st
from textblob import TextBlob

# Title
st.title("Sentiment Analysis App")

# Text input
user_input = st.text_area("Enter your text:")

# Perform sentiment analysis when user enters text
if user_input:
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity

    # Determine sentiment category
    if sentiment > 0:
        st.write("Sentiment: Positive 😊")
    elif sentiment < 0:
        st.write("Sentiment: Negative 😠")
    else:
        st.write("Sentiment: Neutral 😐")

