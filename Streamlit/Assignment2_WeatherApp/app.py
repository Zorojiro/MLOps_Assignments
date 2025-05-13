import streamlit as st
import requests

# Streamlit app title
st.title("Weather Information App")

# User input for city
city = st.text_input("Enter the city name", "London")

# Fetch weather information using an API
if st.button("Get Weather"):
    try:
        # OpenWeatherMap API (replace with your API key)
        api_key = "7ee6067ef28c62f99cd5fa08c42c091a"
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        response = requests.get(base_url, params=params)
        data = response.json()

        # Display weather information
        if response.status_code == 200:
            st.write(f"### Weather in {city}:")
            st.write(f"- **Temperature:** {data['main']['temp']} °C")
            st.write(f"- **Weather:** {data['weather'][0]['description'].capitalize()}")
            st.write(f"- **Humidity:** {data['main']['humidity']}%")
            st.write(f"- **Wind Speed:** {data['wind']['speed']} m/s")
        else:
            st.error("City not found. Please check the name.")
    except Exception as e:
        st.error("Error fetching weather data.")
