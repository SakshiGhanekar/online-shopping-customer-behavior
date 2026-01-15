import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Customer Behavior", layout="centered")

st.title("🛒 Online Shopping Customer Behavior Study")

# Load dataset
df = pd.read_csv("online_shoppers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Convert Revenue to numeric if needed
if df['Revenue'].dtype != 'int64':
    df['Revenue'] = df['Revenue'].map({True: 1, False: 0})

st.subheader("Purchase vs No Purchase")

fig, ax = plt.subplots()
sns.countplot(x='Revenue', data=df, ax=ax)
ax.set_xlabel("Revenue (0 = No, 1 = Yes)")
ax.set_ylabel("Count")
st.pyplot(fig)

st.success("App loaded successfully 🚀")

