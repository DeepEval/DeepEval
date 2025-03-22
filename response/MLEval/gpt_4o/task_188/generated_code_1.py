from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data (you should replace these with your actual training data)
X_train = np.random.rand(100, 5)  # 100 samples, 5 features
Y_train = np.random.rand(100)

def method():
    # Define the model
    model = RandomForestRegressor(random_state=42)

    # Define hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, Y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Optionally, evaluate the best model on the training data (or a validation set if available)
    Y_train_pred = best_model.predict(X_train)
    mse = mean_squared_error(Y_train, Y_train_pred)

    # Output: Best model and its MSE on the training set
    output = {
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'train_mse': mse
    }

    return output

# Call the method for validation
optimized_model_info = method()
print(optimized_model_info)