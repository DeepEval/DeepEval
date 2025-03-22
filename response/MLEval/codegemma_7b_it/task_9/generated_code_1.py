from sklearn.linear_model import LinearRegression

def method():
    # Load the data
    data = pd.read_csv("your_data.csv")

    # Separate the features and target variable
    X = data["days"]
    y = data["morale"]

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X.values.reshape(-1, 1), y)

    # Predict the morale for a new data point
    new_data = pd.DataFrame({"days": [10]})
    prediction = model.predict(new_data["days"].values.reshape(-1, 1))

    # Return the prediction
    return prediction

# Call the method to generate the prediction
prediction = method()

# Print the prediction
print(prediction)