import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data (for demonstration purposes)
np.random.seed(0)
X1 = np.random.rand(100, 1)  # 100 samples, single feature
Y = 4.5 * X1.squeeze() + np.random.randn(100)  # target variable with some noise

def method():
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.2, random_state=42)
    
    # Initialize and fit the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    # Predict on the test set
    Y_pred = model.predict(X_test)
    
    # Calculate the coefficients
    mommy_coefficient = model.coef_[0]
    dadmy_intercept = model.intercept_
    
    # Calculate mean squared error for evaluation
    mse = mean_squared_error(Y_test, Y_pred)
    
    # Prepare output
    output = {
        'MOMMY Coefficient': mommy_coefficient,
        'DADMY Intercept': dadmy_intercept,
        'Mean Squared Error': mse,
        'R-squared': model.score(X_test, Y_test)
    }
    
    return output

# Call the method for validation
output = method()
print(output)