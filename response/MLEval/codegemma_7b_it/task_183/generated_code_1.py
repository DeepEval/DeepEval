# Import necessary packages
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create an Extra Trees classifier
model = ExtraTreesClassifier()

# Fit the model to the data
model.fit(iris.data, iris.target)

# Predict the labels for the test data
y_pred = model.predict(iris.data)

# Print the accuracy of the model
print("Accuracy:", model.score(iris.data, iris.target))

# Define the 'method()' function
def method():

    # Create an Extra Trees classifier
    model = ExtraTreesClassifier()

    # Fit the model to the data
    model.fit(iris.data, iris.target)

    # Predict the labels for the test data
    y_pred = model.predict(iris.data)

    # Return the accuracy of the model
    return model.score(iris.data, iris.target)

# Call the 'method()' function and print the result
print(method())