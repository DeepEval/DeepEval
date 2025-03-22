import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Define the method to fit the Keras model
def method():

    # Load the training data (replace with your actual data loading logic)
    X_train_indices = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    Y_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Convert the target labels to one-hot encoded vectors
    Y_train_oh = to_categorical(Y_train)

    # Create the Keras model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=1))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32)

    # Return the final output (optional)
    return model.evaluate(X_train_indices, Y_train_oh)

# Call the method for validation
method()