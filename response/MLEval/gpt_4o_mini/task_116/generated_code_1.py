import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def method():
    # Generating sample data
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10  # Feature variable
    y = np.random.rand(100) * 100     # Outcome variable

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Creating and training the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting the mean of the outcome variable on the test set
    mean_prediction = model.predict(X_test).mean()

    return mean_prediction

# Calling the method for validation
output = method()
print(f'The predicted mean of the outcome variable is: {output}')