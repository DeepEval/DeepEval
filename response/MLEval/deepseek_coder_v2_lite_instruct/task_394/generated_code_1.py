import numpy as np
from sklearn.linear_model import LinearRegression

def method():
    # Example training data
    X_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y_train = np.dot(X_train, np.array([1, 2])) + 3

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Output the model coefficients (optional)
    output = model.coef_
    return output

# Call the method for validation
output = method()
print(output)