import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Outlier Detection using Z-Score", layout="wide")

st.title("🏠 Outlier Detection using Z-Score")

# Read CSV directly
df = pd.read_csv("AB_NYC_2019.csv")

st.subheader("Dataset")
st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

# Select numeric column
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

column = st.selectbox("Select Numeric Column", numeric_cols)

threshold = st.slider(
    "Select Z-Score Threshold",
    min_value=1.0,
    max_value=5.0,
    value=3.0,
    step=0.1
)

mean = df[column].mean()
std = df[column].std()

z_scores = (df[column] - mean) / std

outliers = df[np.abs(z_scores) > threshold]

st.subheader("Outliers")
st.write("Number of Outliers:", len(outliers))
st.dataframe(outliers)

if st.checkbox("Show Cleaned Dataset"):
    cleaned = df[np.abs(z_scores) <= threshold]
    st.write("Shape:", cleaned.shape)
    st.dataframe(cleaned)

st.subheader("Summary Statistics")
st.write(df[column].describe())
