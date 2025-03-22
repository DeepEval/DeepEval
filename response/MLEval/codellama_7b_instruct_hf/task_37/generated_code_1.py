from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# define the hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'degree': [2, 3, 4]
}

# define the grid search object
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='f1_macro')

# fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# get the best hyperparameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# evaluate the best model on the test data
y_pred = best_model.predict(X_test)
f1_score = f1_score(y_test, y_pred, average='macro')

# print the results
print('Best parameters:', best_params)
print('F1 score:', f1_score)