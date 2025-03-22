from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def method():
    # Define the parameters for the grid search
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # Initialize the grid search object
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1_macro')

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the best score
    print('Best parameters:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)

    # Use the best model to make predictions on the test set
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Evaluate the performance of the best model on the test set
    f1_score_test = f1_score(y_test, y_pred, average='macro')
    print('Test set F1 score:', f1_score_test)

    return y_pred