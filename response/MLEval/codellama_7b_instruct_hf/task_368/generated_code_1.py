import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def method(train_data, test_data, target_variable):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_data, test_data[target_variable], test_size=0.2, random_state=42)

    # Create and fit the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Return the predicted values
    return y_pred

# Example usage
train_data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [6, 7, 8, 9, 10]})
test_data = pd.DataFrame({"feature1": [1.5, 2.5, 3.5, 4.5, 5.5]})
target_variable = "target"

output = method(train_data, test_data, target_variable)
print(output)


train_data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [6, 7, 8, 9, 10]})
test_data = pd.DataFrame({"feature1": [1.5, 2.5, 3.5, 4.5, 5.5]})
target_variable = "target"

output = method(train_data, test_data, target_variable)
print(output)