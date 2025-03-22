from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

def method():
    # Generate some sample data for regression
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Calculate the Mean Squared Error on the test data
    mse = mean_squared_error(y_test, predictions)

    # Output the model and the performance metric
    output = {
        'model': model,
        'mse': mse
    }

    return output

# Call the method to train the model and validate
result = method()
print(f"Trained model: {result['model']}")
print(f"Mean Squared Error on test data: {result['mse']}")