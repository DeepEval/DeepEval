from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def method():
  # Define the model and hyperparameter grid
  model = LogisticRegression(max_iter=1000)
  params = {'C': [0.1, 0.5, 1, 10]}

  # Create the grid search object
  grid_search = GridSearchCV(model, params, cv=5)

  # Fit the grid search to the training data (assuming X_train and y_train are defined)
  grid_search.fit(X_train, y_train)

  # Get the best parameters and score
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_

  # Print the best parameters and score
  print("Best Parameters:", best_params)
  print("Best Score:", best_score)

  # Use the best parameters to fit the final model
  final_model = LogisticRegression(**best_params)
  final_model.fit(X_train, y_train)

  # Return the final model (optional)
  return final_model

# Call the method for validation
method()