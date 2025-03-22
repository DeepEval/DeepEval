import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# Assuming we have a dataset in CSV format and a target variable 'y'
def load_data(file_path):
    # Example: Load data, let's say it's a CSV file with columns 'A' and 'B'
    df = pd.read_csv(file_path)
    X = df[['A']]  # Features
    y = df['B']     # Target variable
    return X, y

def method():
    # Load data
    # X, y = load_data('path_to_your_data.csv')

    # 修改为本地数据文件
    X, y = load_data('evaluation/dynamic_checking/baselines/MLEval/deepseek_v2_lite_chat/testcases/task_368.csv')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Define the parameter grid for tuning (optional)
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['mse', 'mae']
    }

    # Grid search for tuning parameters
    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # Fit the model with best parameters
    regressor.set_params(**grid_search.best_params_)
    regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error: ", mse)

    # Plot feature importance
    importance = regressor.feature_importances_
    plt.figure()
    plt.title("Feature Importances")
    plt.barh(X.columns, importance, color='b', align='best')
    plt.xlabel("Relative Importance")
    plt.show()

    # Return the output
    return y_pred

# Call the method for validation
output = method()