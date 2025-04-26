import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd  # Added missing import for pandas

# Set the page configuration of the app, including the page title, icon, and layout.
st.set_page_config(page_title="Timelytics Forecasting App", page_icon="📦", layout="centered")

# Display the title and captions for the app.
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times. By combining the strengths of these three algorithms, Timelytics provides a robust and reliable prediction of OTD times, helping businesses to optimize their supply chain operations."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them, reducing lead times and improving delivery times. The model utilizes historical data on order processing times, production lead times, shipping times, and other relevant variables to generate accurate forecasts of OTD times. These forecasts can be used to optimize inventory management, improve customer service, and increase overall efficiency in the supply chain."
)


# Load the trained ensemble model from the saved pickle file.
modelfile = "voting_model.pkl"

@st.cache_resource
def load_model():
    with open(modelfile, "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model

voting_model = load_model()

# Define the function for the wait time predictor using the loaded model.
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    prediction = voting_model.predict(
        np.array(
            [
                [
                    purchase_dow,
                    purchase_month,
                    year,
                    product_size_cm3,
                    product_weight_g,
                    geolocation_state_customer,
                    geolocation_state_seller,
                    distance,
                ]
            ]
        )
    )
    return round(prediction[0])

# Define the input parameters using Streamlit's sidebar.
with st.sidebar:
    img = Image.open("/assets/supply_chain_optimisation.jpg")
    st.image(img, use_column_width=True)
    st.header("Input Parameters")
    purchase_dow = st.number_input(
        "Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3
    )
    purchase_month = st.number_input(
        "Purchased Month", min_value=1, max_value=12, step=1, value=1
    )
    year = st.number_input("Purchased Year", min_value=2000, max_value=2030, step=1, value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", min_value=0, value=9328)
    product_weight_g = st.number_input("Product Weight in grams", min_value=0, value=1800)
    geolocation_state_customer = st.number_input(
        "Geolocation State of the Customer", min_value=0, value=10
    )
    geolocation_state_seller = st.number_input(
        "Geolocation State of the Seller", min_value=0, value=20
    )
    distance = st.number_input("Distance (km)", min_value=0.0, value=475.35)

    submit = st.button("Predict Wait Time")

# Define the submit button for the input parameters.
with st.container():
    st.header("Output: Wait Time in Days")

    # When the submit button is clicked, call the wait time predictor function and display the predicted wait time.
    if submit:
        with st.spinner(text="Predicting wait time..."):
            prediction = waitime_predictor(
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance,
            )
            st.success(f"Predicted Wait Time: {prediction} days")

    # Define a sample dataset for demonstration purposes.
    data = {
        "Purchased Day of the Week": [0, 3, 1],
        "Purchased Month": [6, 3, 1],
        "Purchased Year": [2018, 2017, 2018],
        "Product Size in cm³": [37206.0, 63714.0, 54816.0],
        "Product Weight in grams": [16250.0, 7249.0, 9600.0],
        "Geolocation State Customer": [25, 25, 25],
        "Geolocation State Seller": [20, 7, 20],
        "Distance": [247.94, 250.35, 4.915],
    }

    # Create a DataFrame from the sample dataset.
    df = pd.DataFrame(data)

    # Display the sample dataset in the Streamlit app.
    st.header("Sample Dataset")
    st.dataframe(df)
