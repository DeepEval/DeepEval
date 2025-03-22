import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def method():
    # Generate some sample data
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = 2.5 * X + np.random.randn(100, 1)  # Linear relationship with some noise

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Calculate the mean squared error as an evaluation metric
    mse = mean_squared_error(y_test, predictions)

    # Return the predictions and the mean squared error
    output = {
        "predictions": predictions,
        "mean_squared_error": mse
    }
    
    return output

# Call the method for validation
output = method()
print(output)