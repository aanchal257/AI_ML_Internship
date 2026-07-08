import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df=pd.read_csv("employee_retention.csv")

#Page Configuration
st.set_page_config(page_title="Employee Retention App",layout ="centered")

# Sidebar
st.sidebar.title("Employee Retention App")
st.sidebar.info("""
This application predicts whether an employee will stay or leave.

Model Used:
- Logistic Regression
""")

#Title
st.title("Employee Retention Prediction App")

#Overview
st.write(""" ### Overview
This application predicts whether an employee will stay in the company or leave based on working conditions like salary, satisfaction level, working hours, and experience.
""")

#Objective
st.write("""
### Objective
- Analyze employee behavior
- Identify reasons for employee attrition
- Build a machine learning model for prediction
""")

st.markdown("---")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])


#Explore Dataset
st.subheader("Dataset Information")

st.write(df.describe())

st.write("Column Names")
st.write(df.columns.tolist())

st.write("Missing Values")
st.write(df.isnull().sum())


#Data Cleaning
st.subheader("Data Cleaning")

st.write("Duplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

st.write("Data Types")
st.write(df.dtypes)

st.markdown("---")
st.header("Exploratory Data Analysis")


#EDA
print(df.left.value_counts())
print(df.groupby('left')['satisfaction_level'].mean())
print(df.groupby('left')['average_montly_hours'].mean())
print(df.groupby('left')['number_project'].mean())
print(df.groupby('left')['salary'].value_counts())
print(df.groupby('left')['Department'].value_counts())
print(df.corr(numeric_only=True))

#Graph 1 (Salary vs Employee Retention)
st.subheader("Salary vs Employee Retention")

st.write("""
This bar chart shows the relationship between employees' salary levels and whether they stayed or left the company.
""")

fig, ax = plt.subplots()

pd.crosstab(df.salary, df.left).plot(kind="bar", ax=ax)

ax.set_title("Salary vs Employee Retention")
ax.set_xlabel("Salary")
ax.set_ylabel("Number of Employees")

st.pyplot(fig)
st.pyplot(plt)
plt.clf()

#Graph 2 (Department vs Employee Retention)
st.subheader("Department-wise Employee Retention")

st.write("""
This chart compares employee retention across different departments, helping identify departments with higher attrition.
""")

fig, ax = plt.subplots()

pd.crosstab(df.Department, df.left).plot(kind="bar", ax=ax)

ax.set_title("Department vs Employee Retention")
ax.set_xlabel("Department")
ax.set_ylabel("Number of Employees")

st.pyplot(fig)
st.pyplot(plt)
plt.clf()


#Count Plot 1(Salary vs Employee Retention)
sns.countplot(x='salary', data=df)
plt.title("Employee Count by Salary")
plt.savefig("images/salary_countplot.png")
st.pyplot(plt)
plt.clf()

#Count Plot 2(Department vs Employee Retention)
sns.countplot(x='Department', data=df)
plt.title("Employee Count by Department")
plt.xticks(rotation=45)
plt.savefig("images/department_countplot.png")
st.pyplot(plt)
plt.clf()

#Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap_countplot.png")
st.pyplot(plt)
plt.clf()

#Feature Selection
X = df[['satisfaction_level',
        'average_montly_hours',
        'promotion_last_5years',
        'Department',
        'salary']]

y=df['left']

st.subheader("Selected Features")

st.dataframe(X.head())


print("\nTarget (y):")
print(y.head())

#Encoding
department_dummies =pd.get_dummies(df['Department'],prefix='Department')
salary_dummies =pd.get_dummies(df['salary'],prefix='salary')
X=pd.concat([X,department_dummies,salary_dummies], axis=1)
X=X.drop(['Department','salary'], axis=1)

print(X.head())
print("Updated Features:")
print(X.head())

#Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

st.write("Training Data Shape:", X_train.shape)

st.write("Testing Data Shape:", X_test.shape)

#Create Logistic Regression Model
model= LogisticRegression()

#Train the Model
model.fit(X_train,y_train)

#Predict
y_pred= model.predict(X_test)

print("Predicted Values:")
print(y_pred)

#Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
st.markdown("---")
st.header("Model Performance")
st.metric("Accuracy", f"{accuracy*100:.2f}%")

con_matrix= confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(con_matrix,
            annot=True,
            fmt='d',
            cmap='Blues')

st.pyplot(fig)

class_matrix=classification_report(y_test, y_pred)
st.subheader("Classification Report")
st.text(class_matrix)

st.markdown("---")
st.caption("Employee Retention Prediction App | Built using Streamlit and Logistic Regression")

st.markdown("---")
st.header("Predict Employee Retention")

satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)

hours = st.slider("Average Monthly Hours", 90, 320, 200)

promotion = st.selectbox("Promotion in Last 5 Years", [0,1])

department = st.selectbox(
    "Department",
    sorted(df["Department"].unique())
)

salary = st.selectbox(
    "Salary",
    ["low","medium","high"]
)