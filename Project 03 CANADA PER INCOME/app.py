import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Canada Per Capita Income Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Canada Per Capita Income Prediction using Linear Regression")

st.write("""
This application predicts Canada's **Per Capita Income**
using **Linear Regression**.
""")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("canada_per_capita_income (1).csv")

st.subheader("Dataset")

st.dataframe(df)

# -------------------------------
# Prepare Data
# -------------------------------
X = df[['year']]
y = df['per capita income (US$)']

# -------------------------------
# Train Model
# -------------------------------
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# Show Model Parameters
# -------------------------------
st.subheader("Model Information")

st.write("Slope (Coefficient):", model.coef_[0])
st.write("Intercept:", model.intercept_)

# -------------------------------
# User Input
# -------------------------------
st.subheader("Predict Income")

year = st.number_input(
    "Enter Year",
    min_value=1970,
    max_value=2100,
    value=2020,
    step=1
)

prediction = model.predict([[year]])

st.success(f"Predicted Per Capita Income for {year}: ${prediction[0]:,.2f}")

# -------------------------------
# Plot Regression
# -------------------------------
st.subheader("Regression Plot")

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(X, y, color='blue', label="Actual Data")
ax.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")

ax.scatter(
    year,
    prediction[0],
    color='green',
    s=120,
    label="Prediction"
)

ax.set_xlabel("Year")
ax.set_ylabel("Per Capita Income (US$)")
ax.set_title("Canada Per Capita Income Prediction")
ax.legend()

st.pyplot(fig)

# -------------------------------
# Predict 2020
# -------------------------------
st.subheader("Prediction for Year 2020")

income2020 = model.predict([[2020]])

st.info(f"Predicted Income in 2020 = ${income2020[0]:,.2f}")
