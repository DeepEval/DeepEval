import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def method():
    # 1. Load your data (replace 'your_data.csv' with your actual file)
    # data = pd.read_csv('your_data.csv') 

    # 修改为本地数据文件
    data = pd.read_csv('evaluation/dynamic_checking/baselines/MLEval/gemma_2_9b_it/testcases/task_410.csv')

    # 2. Separate features (X) and target variable (y)
    X = data.drop('target_variable', axis=1)  # Replace 'target_variable' with your target column name
    y = data['target_variable']

    # 3. Initialize your best model (e.g., Logistic Regression)
    model = LogisticRegression() 

    # 4. Define the parameter grid for grid search
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'] 
    }

    # 5. Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)

    # 6. Get the best model from grid search
    best_model = grid_search.best_estimator_

    # 7. Evaluate the best model (e.g., on a separate test set)
    # ... (code to load test data and evaluate performance)

    # 8. Return the best model
    return best_model

# Call the method to perform grid search and get the best model
best_model = method()

# Print the best model's parameters
print(best_model.get_params())