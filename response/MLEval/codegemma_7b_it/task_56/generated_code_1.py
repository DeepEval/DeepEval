from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def method():

  # Load the iris dataset
  iris = load_iris()

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2)

  # Create a Logistic Regression model
  model = LogisticRegression()

  # Train the model on the training data
  model.fit(X_train, y_train)

  # Predict the labels for the testing data
  y_pred = model.predict(X_test)

  # Return the predictions
  return y_pred

# Call the method and print the results
predictions = method()
print(predictions)