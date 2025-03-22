import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def method():
    # Load your dataset (replace 'your_dataset.csv' with your file)
    # data = pd.read_csv('your_dataset.csv')
    
    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_368.csv')

    # Split data into features (X) and target variable (y)
    X = data.drop('target_variable', axis=1)  # Replace 'target_variable' with your target column name
    y = data['target_variable']

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest Regressor
    rf_regressor = RandomForestRegressor()

    # Define the hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Return the best model
    return best_model

# Call the method to run the code
best_model = method()