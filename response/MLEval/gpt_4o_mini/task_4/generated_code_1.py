import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def method():
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = 3 * X.squeeze() + np.random.randn(100) * 3  # Linear relation with noise

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create and fit the Ridge regression model
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)

    # Make predictions
    predictions = ridge_model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)

    return mse

# Call the method for validation
output = method()
print(f'Mean Squared Error: {output}')