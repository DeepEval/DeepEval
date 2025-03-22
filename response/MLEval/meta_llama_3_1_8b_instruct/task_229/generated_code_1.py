# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

def method():
    """
    This function trains a simple neural network model and uses the X_test, y_test 
    from the train_test_split step for the validation_data parameter.
    """

    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Convert target variable to categorical
    y = to_categorical(y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and compile the neural network model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(4,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Return the model's performance on the test data
    return model.evaluate(X_test, y_test)

# Call the method to validate the model
output = method()
print(output)