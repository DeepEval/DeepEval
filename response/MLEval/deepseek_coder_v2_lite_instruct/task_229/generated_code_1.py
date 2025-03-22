import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def method():
    # Assuming X and y are defined somewhere above this function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Validate the model
    score = model.score(X_test, y_test)

    return score

# Example usage
if __name__ == "__main__":
    # Assuming X and y are defined here for the sake of example
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])

    output = method()
    print("Model's score on test data:", output)