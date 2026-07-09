import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Outlier Detection using Z-Score",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Outlier Detection using Z-Score")
st.write("This application demonstrates different techniques for detecting and removing outliers.")

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload bhp.csv Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.header("Original Dataset")

    st.dataframe(df.head())

    # ---------------------------------------------------
    # Find Numerical Column
    # ---------------------------------------------------

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    st.sidebar.header("Select Column")

    column = st.sidebar.selectbox(
        "Choose Numerical Column",
        numeric_columns
    )

    # ---------------------------------------------------
    # Dataset Statistics
    # ---------------------------------------------------

    st.header("Dataset Statistics")

    st.write(df[column].describe())

    # ---------------------------------------------------
    # Histogram
    # ---------------------------------------------------

    st.header("Histogram")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(df[column], bins=40)

    ax.set_xlabel(column)

    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # ---------------------------------------------------
    # Histogram (Log Scale)
    # ---------------------------------------------------

    st.header("Histogram (Log Scale)")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(df[column], bins=40)

    ax.set_yscale("log")

    ax.set_xlabel(column)

    ax.set_ylabel("Frequency (Log Scale)")

    st.pyplot(fig)

    # ---------------------------------------------------
    # Remove Outliers using Percentiles
    # ---------------------------------------------------

    st.header("Step 1 : Remove Outliers using Percentiles")

    lower_limit = df[column].quantile(0.001)

    upper_limit = df[column].quantile(0.999)

    df2 = df[
        (df[column] >= lower_limit) &
        (df[column] <= upper_limit)
    ]

    st.write("Original Rows :", len(df))

    st.write("Rows after Percentile Method :", len(df2))

    st.dataframe(df2.head())

    # ---------------------------------------------------
    # Remove using Standard Deviation
    # ---------------------------------------------------

    st.header("Step 2 : Remove Outliers using Standard Deviation")

    mean = df2[column].mean()

    std = df2[column].std()

    lower = mean - 4 * std

    upper = mean + 4 * std

    df3 = df2[
        (df2[column] >= lower) &
        (df2[column] <= upper)
    ]

    st.write("Rows after Standard Deviation :", len(df3))

    st.dataframe(df3.head())

    # ---------------------------------------------------
    # Remove using Z Score
    # ---------------------------------------------------

    st.header("Step 3 : Remove Outliers using Z-Score")

    df4 = df2.copy()

    df4["Z-Score"] = zscore(df4[column])

    final_df = df4[df4["Z-Score"].abs() <= 4]

    st.write("Rows after Z-Score :", len(final_df))

    st.dataframe(final_df.head())

    # ---------------------------------------------------
    # Comparison
    # ---------------------------------------------------

    st.header("Comparison")

    comparison = pd.DataFrame({

        "Method":[
            "Original",
            "Percentile",
            "Standard Deviation",
            "Z-Score"
        ],

        "Rows":[
            len(df),
            len(df2),
            len(df3),
            len(final_df)
        ]
    })

    st.table(comparison)

    # ---------------------------------------------------
    # Final Histogram
    # ---------------------------------------------------

    st.header("Final Cleaned Dataset Histogram")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(final_df[column], bins=40)

    ax.set_xlabel(column)

    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # ---------------------------------------------------
    # Download Cleaned Dataset
    # ---------------------------------------------------

    st.header("Download Cleaned Dataset")

    csv = final_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

else:

    st.info("Please upload the BHP dataset (bhp.csv).")
