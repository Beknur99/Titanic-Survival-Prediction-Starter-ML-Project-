# Titanic-Survival-Prediction-Starter-ML-Project-
# Description

This project builds a simple machine learning model that predicts whether a passenger survived the Titanic disaster using basic demographic and ticket data.
It’s designed as an introductory ML project — easy to understand, but demonstrates the full workflow from data preprocessing to model evaluation.

# Features

Loads and cleans the Titanic dataset from Kaggle.

Encodes categorical features (e.g., gender).

Handles missing values (e.g., median age).

Splits the data into training and testing sets.

Trains a Random Forest Classifier.

Evaluates accuracy on unseen data.

Includes a real-time example prediction.

# Technologies Used

Python 3

pandas — data manipulation

scikit-learn — model training and evaluation

# Model Info

Algorithm used: Random Forest Classifier
Train/Test split: 80% / 20%

Feature	Description
Pclass	Passenger class (1 = First, 3 = Third)
Sex	0 = Male, 1 = Female
Age	Age in years
Fare	Ticket price
