from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # Load example data (replace with your dataset)
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select the model (replace with your best model)
    model = RandomForestClassifier(random_state=42)

    # Define the parameter grid (customize based on the model you choose)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions and evaluate
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Output the best model, parameters, and accuracy
    output = {
        'best_model': best_model,
        'best_params': best_params,
        'accuracy': accuracy
    }
    
    return output

# Call the method for validation
result = method()
print("Best Model:", result['best_model'])
print("Best Parameters:", result['best_params'])
print("Accuracy:", result['accuracy'])