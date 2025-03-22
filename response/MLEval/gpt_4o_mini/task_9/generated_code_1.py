import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def method():
    # Sample data: number of days and corresponding morale scores
    days = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    morale = np.array([5, 6, 7, 8, 7, 8, 9, 10, 11, 12])  # Hypothetical morale scores

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(days, morale, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Calculate the output
    output = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'predictions': predictions,
        'actual': y_test
    }

    # Optional: Plotting the results for visualization
    plt.scatter(days, morale, color='blue', label='Actual morale')
    plt.plot(days, model.predict(days), color='red', label='Fitted line')
    plt.xlabel('Days')
    plt.ylabel('Morale')
    plt.title('Linear Regression: Morale vs. Days')
    plt.legend()
    plt.show()

    return output

# Call the method for validation
result = method()
print(result)