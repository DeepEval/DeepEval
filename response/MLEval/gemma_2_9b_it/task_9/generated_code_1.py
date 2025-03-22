import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def method():
    # Assuming your data is in a pandas DataFrame called 'df'
    # with columns 'days' and 'morale'

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['days'], df['morale'], test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train.values.reshape(-1, 1), y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test.values.reshape(-1, 1))

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # You can return the model, predictions, or other relevant information here
    output = {'model': model, 'predictions': y_pred, 'mse': mse}
    return output

# Call the method to validate
result = method()