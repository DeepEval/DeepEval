from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def method():
    # Create a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize a RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    
    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    output = (best_params, best_score)
    return output

# Call the method function for validation
best_parameters, best_score = method()
print("Best Parameters:", best_parameters)
print("Best Cross-Validation Score:", best_score)