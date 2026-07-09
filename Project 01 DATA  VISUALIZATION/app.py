import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Page Configuration
st.set_page_config(
    page_title="Google Play Store Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Google Play Store Data Analysis Dashboard")

st.markdown("""
Analyze Google Play Store applications using interactive visualizations.
""")

#Load csv
@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore.csv")
    return df

df = load_data()

#Sidebar
st.sidebar.title("Navigation")

option = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset",
        "Data Cleaning",
        "EDA",
        "Insights",
        "About"
    ]
)

#Home
if option=="Home":

    st.header("Project Overview")

    col1,col2,col3=st.columns(3)

    col1.metric("Rows",df.shape[0])

    col2.metric("Columns",df.shape[1])

    col3.metric("Missing Values",df.isnull().sum().sum())

    st.write(df.head())

#Dataset
elif option=="Dataset":

    st.header("Dataset")

    st.dataframe(df)

    st.subheader("Data Types")

    st.write(df.dtypes)

    st.subheader("Missing Values")

    st.write(df.isnull().sum())


#Data Cleaning
elif option=="Data Cleaning":

    st.header("Data Cleaning")

    st.write("Missing Values")

    st.write(df.isnull().sum())

    st.write("Duplicate Records")

    st.write(df.duplicated().sum())

    # Remove duplicate records
    df = df.drop_duplicates()

    st.success("Duplicate records removed successfully!")

#EDA
elif option=="EDA":

    st.header("Exploratory Data Analysis")

    st.subheader("Top Categories")

    fig,ax=plt.subplots(figsize=(10,6))

    df["Category"].value_counts().head(10).plot(kind="bar",ax=ax)

    st.pyplot(fig)


    #Histogram
    st.subheader("Ratings")

    fig,ax=plt.subplots()

    df["Rating"].dropna().hist(ax=ax)

    st.pyplot(fig)

    #Pie Chart
    st.subheader("Content Rating")

    fig,ax=plt.subplots()

    df["Content Rating"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    st.pyplot(fig)

    #Heatmap
    st.subheader("Correlation")

    fig,ax=plt.subplots(figsize=(8,5))

    sns.heatmap(df.corr(numeric_only=True),annot=True,ax=ax)

    st.pyplot(fig)

#Insights
elif option=="Insights":

    st.header("Insights")

    st.write("""
    ✔ Most apps are Free.

    ✔ Family category contains maximum apps.

    ✔ Ratings mostly lie between 4 and 4.5.

    ✔ Higher reviews generally indicate more installs.

    ✔ Free apps dominate the Play Store.
    """)

