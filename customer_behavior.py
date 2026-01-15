# STEP 4: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 5: Load dataset
df = pd.read_csv("online_shoppers.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# STEP 6: Data Cleaning
# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing numeric values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Convert target column Revenue to numeric
df['Revenue'] = df['Revenue'].map({True: 1, False: 0})

print("\nAfter cleaning:")
print(df.isnull().sum())

# STEP 7: Exploratory Data Analysis (EDA)

# 1. Purchase vs No Purchase
plt.figure(figsize=(5,4))
sns.countplot(x='Revenue', data=df)
plt.title("Purchase vs No Purchase")
plt.xlabel("Revenue (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# 2. Page Value vs Purchase
plt.figure(figsize=(6,4))
sns.boxplot(x='Revenue', y='PageValues', data=df)
plt.title("Page Value vs Purchase")
plt.show()

# 3. Visitor Type vs Purchase
plt.figure(figsize=(6,4))
sns.countplot(x='VisitorType', hue='Revenue', data=df)
plt.title("Visitor Type vs Purchase")
plt.xticks(rotation=20)
plt.show()

# STEP 8: Simple Machine Learning Model (Optional but Recommended)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Convert categorical columns to numeric
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('Revenue', axis=1)
y = df_encoded['Revenue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 9: Final Message
print("\nProject Completed Successfully!")

