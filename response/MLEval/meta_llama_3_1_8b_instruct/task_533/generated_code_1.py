# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest

# Generate a sample dataset for demonstration
np.random.seed(0)
X = np.random.rand(100, 1)
y = 3 + 2 * X + np.random.randn(100, 1) * 0.1

# Combine X and y into a DataFrame
df = pd.DataFrame(np.hstack((X, y)), columns=['X', 'y'])

# Split the dataset into features (X) and target (y)
X = df[['X']]
y = df['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the residuals
residuals = y_test - y_pred

# Calculate the distances from the mean
distances = np.abs(residuals - residuals.mean())

# Identify outliers using Isolation Forest
iforest = IsolationForest(n_estimators=100, random_state=42)
iforest.fit(X_train)
outlier_mask = iforest.predict(X_train) == -1

# Remove outliers and high leverage points
X_train_cleaned = X_train[~outlier_mask]

# Re-fit the model with the cleaned data
model_cleaned = LinearRegression()
model_cleaned.fit(X_train_cleaned, y_train)

# Make predictions with the cleaned model
y_pred_cleaned = model_cleaned.predict(X_test)

# Calculate the mean squared error of the cleaned model
mse_cleaned = mean_squared_error(y_test, y_pred_cleaned)

# Print the results
print("Original model's mean squared error:", mean_squared_error(y_test, y_pred))
print("Cleaned model's mean squared error:", mse_cleaned)

# Define the method function
def method():
    return mse_cleaned

# Call the method function for validation
output = method()
print("The final output is:", output)