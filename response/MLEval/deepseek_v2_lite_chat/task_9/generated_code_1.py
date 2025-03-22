import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def method():
    # Sample data
    data = {
        'Days': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Morale': [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]
    }
    df = pd.DataFrame(data)

    # Split the data into features (X) and target (y)
    X = df['Days'].values.reshape(-1, 1)
    y = df['Morale'].values.reshape(-1, 1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict morale on the test set
    y_pred = model.predict(X_test)

    # Plot the data and the regression line
    plt.scatter(X, y, color='red')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.title('Morale vs. Number of Days')
    plt.xlabel('Days')
    plt.ylabel('Morale')
    plt.grid(True)
    plt.show()

    # Get the slope and intercept of the regression line
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    # Output the predicted morale for a given number of days
    def predict_morale(days):
        return slope * days + intercept

    return predict_morale

# Call the method for validation
output = method()
print("Predicted morale for 15 days:", output(15))