import numpy as np
from sklearn.linear_model import LinearRegression

def method():
    # Generate some sample data
    days = np.linspace(0, 100, 1000)
    morale = np.sin(days) + np.random.normal(size=1000)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(days, morale, test_size=0.2, random_state=42)

    # Create a linear regression model and fit the data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the performance of the model
    mse = model.score(X_test, y_test)
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"Mean squared error: {mse}")
    print(f"Mean absolute error: {mae}")

    return model