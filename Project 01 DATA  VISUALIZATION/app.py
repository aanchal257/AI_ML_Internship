import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Google Play Store Analysis", layout="wide")

st.title("📱 Google Play Store Data Visualization")
st.write("Interactive dashboard for exploring Google Play Store apps.")

# Upload dataset
uploaded_file = st.file_uploader("Upload googleplaystore.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Data Cleaning
    df = df.dropna(subset=["Rating"])

    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    df["Installs"] = (
        df["Installs"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
    )
    df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df["Size"] = (
        df["Size"]
        .astype(str)
        .str.replace("M", "", regex=False)
        .str.replace("k", "", regex=False)
        .str.replace("Varies with device", "", regex=False)
    )
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")

    st.sidebar.header("Filters")

    category = st.sidebar.selectbox(
        "Category",
        ["All"] + sorted(df["Category"].dropna().unique().tolist())
    )

    if category != "All":
        df = df[df["Category"] == category]

    st.subheader("Dataset")
    st.dataframe(df)

    st.metric("Total Apps", len(df))
    st.metric("Average Rating", round(df["Rating"].mean(), 2))

    st.divider()

    st.subheader("Rating Distribution")

    fig, ax = plt.subplots()
    ax.hist(df["Rating"], bins=20)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Top Categories")

    fig, ax = plt.subplots(figsize=(8,5))
    df["Category"].value_counts().head(10).plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("Reviews vs Rating")

    fig, ax = plt.subplots()
    ax.scatter(df["Reviews"], df["Rating"])
    ax.set_xlabel("Reviews")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

    st.subheader("Installs vs Rating")

    fig, ax = plt.subplots()
    ax.scatter(df["Installs"], df["Rating"])
    ax.set_xlabel("Installs")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

    st.subheader("Free vs Paid Apps")

    fig, ax = plt.subplots()
    df["Type"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

    st.subheader("Top 10 Highest Rated Apps")

    st.dataframe(
        df.sort_values("Rating", ascending=False)[
            ["App","Category","Rating","Reviews","Installs"]
        ].head(10)
    )

else:
    st.info("Please upload googleplaystore.csv")
