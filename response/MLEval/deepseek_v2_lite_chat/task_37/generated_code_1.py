import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def method():
    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model
    model = SVC()
    
    # Define the hyperparameters
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    
    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    
    # Train the model with the best hyperparameters
    best_model = SVC(**best_params)
    best_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot the performance of the model with different hyperparameters
    plt.figure(figsize=(10, 5))
    plt.plot(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], marker='o')
    plt.xlabel('Hyperparameters')
    plt.ylabel('Mean Cross-Validation Score')
    plt.title('Model Performance for Different Hyperparameters')
    plt.show()
    
    return best_model

# Call the method for validation
best_model = method()