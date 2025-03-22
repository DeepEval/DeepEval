import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def method():
    # Generate some synthetic data for demonstration
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = 2.5 * X + np.random.randn(100, 1) * 2  # Linear relation with noise

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = LinearRegression()

    # Fit the model using the training data
    model.fit(X_train, y_train)

    # Optionally, you can return the trained model or its predictions on test data
    output = model  # or model.predict(X_test) for predictions
    return output

# Call the method for validation
trained_model = method()
print("Model training complete. Model coefficients:", trained_model.coef_)