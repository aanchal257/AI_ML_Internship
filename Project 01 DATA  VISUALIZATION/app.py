import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Google Play Store Analysis", layout="wide")

st.title("📱 Google Play Store Data Visualization")
st.write("Interactive dashboard for exploring the Google Play Store dataset.")

# Load dataset from the project folder
df = pd.read_csv("googleplaystore.csv")

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

# Sidebar Filter
st.sidebar.header("Filters")
category = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(df["Category"].dropna().unique())
)

if category != "All":
    df = df[df["Category"] == category]

# Dataset
st.subheader("Dataset")
st.dataframe(df)

# Metrics
col1, col2 = st.columns(2)
col1.metric("Total Apps", len(df))
col2.metric("Average Rating", round(df["Rating"].mean(), 2))

# Rating Distribution
st.subheader("Rating Distribution")
fig, ax = plt.subplots()
ax.hist(df["Rating"], bins=20)
st.pyplot(fig)

# Top Categories
st.subheader("Top Categories")
fig, ax = plt.subplots(figsize=(8,5))
df["Category"].value_counts().head(10).plot(kind="bar", ax=ax)
st.pyplot(fig)

# Reviews vs Rating
st.subheader("Reviews vs Rating")
fig, ax = plt.subplots()
ax.scatter(df["Reviews"], df["Rating"])
ax.set_xlabel("Reviews")
ax.set_ylabel("Rating")
st.pyplot(fig)

# Free vs Paid Apps
st.subheader("Free vs Paid Apps")
fig, ax = plt.subplots()
df["Type"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
ax.set_ylabel("")
st.pyplot(fig)

# Top Rated Apps
st.subheader("Top 10 Highest Rated Apps")
st.dataframe(
    df.sort_values("Rating", ascending=False)[
        ["App", "Category", "Rating", "Reviews", "Installs"]
    ].head(10)
)
