# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the Boston Housing dataset
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Preprocess the data by scaling the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('PRICE', axis=1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['PRICE'], test_size=0.2, random_state=42)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance using mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Return the predicted prices
output = y_pred

# Call the generated method for validation
def method():
    return output

print(method())