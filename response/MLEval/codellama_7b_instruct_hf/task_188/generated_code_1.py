from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the hyperparameter grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, Y_train)

# Print the best parameters and the corresponding accuracy score
print('Best parameters:', grid_search.best_params_)
print('Best accuracy:', grid_search.best_score_)

# Use the best parameters to train the model on the entire training set
best_model = grid_search.best_estimator_
best_model.fit(X_train, Y_train)

# Evaluate the model on the test set
test_pred = best_model.predict(X_test)
test_acc = accuracy_score(Y_test, test_pred)
print('Test accuracy:', test_acc)