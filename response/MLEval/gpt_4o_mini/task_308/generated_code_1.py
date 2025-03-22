import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def method():
    # Sample data: total request rate and cpu usage (you would replace this with your actual data)
    data = {
        'total_request_rate': [100, 150, 200, 250, 300, 350, 400, 450, 500],
        'cpu': [30, 35, 40, 45, 50, 55, 60, 65, 70],
        'response_time': [1.2, 1.5, 1.7, 1.8, 2.0, 2.3, 2.5, 2.7, 3.0]  # This is what we want to predict
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Features and target variable
    X = df[['total_request_rate', 'cpu']]
    y = df['response_time']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Output the model coefficients and the mean squared error
    output = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'mean_squared_error': mse
    }

    return output

# Call the method for validation
if __name__ == "__main__":
    output = method()
    print(output)