"""
Titanic Survival Prediction (Starter ML Project)
Author: Beknur99
Version: 1.0
A simple machine learning model predicting passenger survival on the Titanic dataset.
"""

# Importing required libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset (train.csv from Kaggle)
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "titanic.csv")
df = pd.read_csv(file_path)

# Display basic info
print("Dataset overview:")
print(df.head())

# Data preprocessing
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})     # convert text to numbers
df['Age'].fillna(df['Age'].median(), inplace=True)      # fill missing ages

# Define features (X) and target (y)
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Split data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Example prediction
example = pd.DataFrame([[3, 0, 25, 15.0]], columns=['Pclass', 'Sex', 'Age', 'Fare'])
result = model.predict(example)[0]
print("\nExample passenger (3rd class, male, 25 years old, 15$ ticket):")
print("Predicted survival:", "Yes" if result == 1 else "No")
