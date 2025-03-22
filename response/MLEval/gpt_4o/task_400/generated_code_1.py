from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def method():
    # For demonstration, we'll generate some example data
    # Let's create a simple linear relationship y = 2x + 1
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Splitting the dataset into training and test sets
    train_set, test_set, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(train_set, train_labels)

    # For demonstration, let's return the coefficients of the trained model
    output = {
        'intercept': model.intercept_,
        'coefficients': model.coef_
    }

    return output

# Call the method function for validation
result = method()
print("Model intercept:", result['intercept'])
print("Model coefficients:", result['coefficients'])