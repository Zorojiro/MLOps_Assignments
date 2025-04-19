from transformers import pipeline
import streamlit as st

senti_model = pipeline("sentiment-analysis")

if senti_model not in st.session_state:
    st.session_state[senti_model] = senti_model

st.title("Sentiment Analysis 😊😥🥲😏")

text = st.text_input("Enter your sentence here...","I love programming!")

if st.button("Analyze"):
    # Perform sentiment analysis
    senti_model = st.session_state[senti_model]

    result = senti_model(text)

    st.markdown("### Sentiment Analysis Result :")

    st.write(f"Sentiment: {result[0]['label']}")
    st.write(f"Score: {result[0]['score']:.2f}")