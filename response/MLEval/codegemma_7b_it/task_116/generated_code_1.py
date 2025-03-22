from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def method():

    # Assuming you have the data loaded as X and y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create a Linear Regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict the mean of the outcome variable for the test data
    y_pred = model.predict(X_test)
    mean_prediction = y_pred.mean()

    return mean_prediction

# Call the method for validation
mean_prediction = method()
print("Mean prediction:", mean_prediction)