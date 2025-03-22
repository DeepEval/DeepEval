import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def method():
    # Example data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 3, 5, 7, 11])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the final output (e.g., mean squared error)
    mse = mean_squared_error(y_test, y_pred)

    # Return the final output
    return mse

# Call the method for validation
output = method()
print("Final Output:", output)