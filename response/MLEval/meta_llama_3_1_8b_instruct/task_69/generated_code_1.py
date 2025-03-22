# Import necessary packages
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Function to create a Keras model
def create_keras_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(4,)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to create a scikit-learn Estimator from a Keras model
def KerasEstimator(model):
    class KerasWrapper(BaseEstimator):
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            self.model.fit(X, y, epochs=10, verbose=0)

        def predict(self, X):
            return self.model.predict(X)

        def score(self, X, y):
            y_pred = self.predict(X)
            return accuracy_score(y, np.argmax(y_pred, axis=1))
    return KerasWrapper(model)

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Keras model
keras_model = create_keras_model()

# Create a scikit-learn Estimator from the Keras model
estimator = KerasEstimator(keras_model)

# Fit the estimator to the training data
estimator.fit(X_train, y_train)

# Evaluate the estimator on the testing data
output = estimator.score(X_test, y_test)

# Method function
def method():
    return output

# Call the generated method for validation
print(method())