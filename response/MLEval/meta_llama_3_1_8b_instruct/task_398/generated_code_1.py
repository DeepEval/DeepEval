# Import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def method():
    # Load Boston housing dataset
    boston = load_boston()
    
    # Convert dataset into DataFrame
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    
    # Split the data into features (X) and the target variable (y)
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize RandomForestRegressor model
    model = RandomForestRegressor()
    
    # Define hyperparameter tuning space
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
       'max_depth': [None, 5, 10, 15, 20],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 5, 10]
    }
    
    # Perform GridSearchCV to tune the model
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Use the best model to make predictions on the test set
    y_pred = best_model.predict(X_test)
    
    # Calculate the mean squared error of the best model
    mse = mean_squared_error(y_test, y_pred)
    
    # Print the best parameters and the mean squared error
    print("Best Parameters:", best_params)
    print("Mean Squared Error:", mse)
    
    # Return the mean squared error as the output
    return mse

# Call the generated method for validation
output = method()
print("Output:", output)