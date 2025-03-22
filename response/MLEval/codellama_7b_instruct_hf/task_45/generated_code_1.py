from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Define the hyperparameters to tune
param_grid = {
    "normalize": [True, False],
    "alpha": [0.001, 0.01, 0.1, 1],
    "fit_intercept": [True, False]
}

# Create a linear regression model
model = LinearRegression()

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(X, y)

# Print the best parameters
print("Best parameters:", grid_search.best_params_)

# Print the best score
print("Best score:", grid_search.best_score_)

# Create a new model with the best parameters
best_model = LinearRegression(**grid_search.best_params_)

# Fit the best model on the entire dataset
best_model.fit(X, y)

# Make predictions on the entire dataset
predictions = best_model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("MSE:", mse)
print("R2:", r2)