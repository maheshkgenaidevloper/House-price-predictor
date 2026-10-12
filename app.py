
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("House Price Prediction")
st.write("Machine Learning using Multiple Linear Regression")

# Load data and train the model
@st.cache_resource
def load_model():
    data = pd.read_csv("house_prices.csv")

    X = data[["Area_sqft", "Bedrooms", "Age_years"]]
    y = data["Price_lakhs"]

    model = LinearRegression()
    model.fit(X, y)

    return model, data

try:
    model, data = load_model()

    st.subheader("Enter House Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input(
            "Area (sq. ft.)",
            min_value=100,
            max_value=10000,
            value=1500
        )

    with col2:
        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=1,
            max_value=10,
            value=3
        )

    with col3:
        age = st.number_input(
            "House Age (years)",
            min_value=0,
            max_value=100,
            value=5
        )

    if st.button("Predict House Price", type="primary"):
        house = pd.DataFrame({
            "Area_sqft": [area],
            "Bedrooms": [bedrooms],
            "Age_years": [age]
        })

        prediction = model.predict(house)[0]

        st.success(
            f"Estimated Price: ₹{prediction:.2f} lakhs"
        )

    st.subheader("Dataset Preview")
    st.dataframe(data, use_container_width=True)

    st.subheader("Dataset Statistics")
    st.dataframe(data.describe(), use_container_width=True)

except FileNotFoundError:
    st.error(
        "house_prices.csv not found. "
        "Place it in the same folder as app.py."
    )