from sklearn.model_selection import GridSearchCV

# Define the model
model = # Your model here

# Define the hyperparameter grid
param_grid = # Your hyperparameter grid here

# Create the GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Get the best score
best_score = grid_search.best_score_

# Get the best parameters
best_params = grid_search.best_params_

# Return the output (optional)
output = best_score, best_params

# Call the method for validation
method()