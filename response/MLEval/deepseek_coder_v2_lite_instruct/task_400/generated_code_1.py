from sklearn.linear_model import LinearRegression

def method():
    # Example data
    train_set = [[1], [2], [3], [4]]
    train_labels = [2, 4, 6, 8]

    # Create and train the model
    model = LinearRegression()
    model.fit(train_set, train_labels)

    # Make a prediction (optional, as per the task requirement)
    output = model.predict([[5]])

    return output

# Call the method for validation
print(method())