# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def method():
    # Define the sequence
    sequence = [1, 2, 4, 7, 11, 16, 22, 29, 37, 46]

    # Split the sequence into input (X) and output (y)
    X = np.arange(len(sequence)).reshape(-1, 1)
    y = np.array(sequence)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make a prediction for the next term
    next_term = model.predict(np.array([[len(sequence)]]))

    # Calculate the Mean Squared Error (MSE) of the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Predicted next term: {next_term[0]}")
    print(f"Mean Squared Error (MSE): {mse}")

    # Return the predicted next term
    return next_term[0]

# Call the method for validation
output = method()
print("The final output is:", output)