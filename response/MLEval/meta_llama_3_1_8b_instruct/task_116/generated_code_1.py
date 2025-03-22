import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 3 + 2 * X + np.random.randn(100, 1)


# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
mae = metrics.mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")


def method():
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.rand(100, 1)
    y = 3 + 2 * X + np.random.randn(100, 1)

    # Preprocess data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Build and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Return the predicted mean of the outcome variable
    return np.mean(y_pred)

output = method()
print(output)