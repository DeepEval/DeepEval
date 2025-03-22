import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

def method():
    # Load your data
    # Replace 'your_data.csv' with your actual data file
    data = pd.read_csv('your_data.csv') 

    # Select your features (X) and target variable (y)
    X = data[['feature1', 'feature2', 'feature3']]  # Replace with your feature columns
    y = data['target_variable']  # Replace with your target variable column

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Calculate Variance Inflation Factor (VIF) for each feature
    vif = pd.DataFrame()
    vif['VIF Factor'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['Features'] = X_train.columns

    # Define a threshold for high VIF (e.g., 5 or 10)
    threshold = 5 

    # Identify high leverage points based on VIF
    high_leverage_features = vif[vif['VIF Factor'] > threshold]['Features'].tolist()

    # Remove high leverage features from the dataset
    X_train_clean = X_train.drop(columns=high_leverage_features)
    X_test_clean = X_test.drop(columns=high_leverage_features)

    # Fit the model again with the cleaned data
    model_cleaned = LinearRegression()
    model_cleaned.fit(X_train_clean, y_train)

    # Evaluate the model performance (e.g., using R-squared, RMSE)
    # ...

    # Compare the results of the original and cleaned models
    # ...

    # Return the output (e.g., model coefficients, performance metrics)
    output = {'original_model_results':  # Results of the original model
              'cleaned_model_results':   # Results of the cleaned model} 

    return output

# Call the method and print the output
results = method()
print(results)