from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Generate a random dataset for demonstration purposes
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Define the parameter grid for the GridSearchCV
param_grid = {'alpha': [0.1, 0.5, 1, 5, 10], 'fit_intercept': [True, False]}

# Create a GridSearchCV object and fit it to the training data
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Print the best parameters and the corresponding score
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

# Return the final output if needed
return grid_search.best_params_