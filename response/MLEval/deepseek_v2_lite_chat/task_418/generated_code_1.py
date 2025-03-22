from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError
import numpy as np

# Assuming we have a dataset that we can use for demonstration
# X_train, X_valid, y_train, y_valid = ...

def method():
    # Create a sequential model
    model = Sequential()
    model.add(Dense(32, input_dim=10, activation='relu'))  # Input layer
    model.add(Dense(64, activation='relu'))                # Hidden layer
    model.add(Dense(32, activation='relu'))                 # Hidden layer
    model.add(Dense(1, activation='linear'))               # Output layer

    # Compile the model with a suitable optimizer and loss function
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=[MeanAbsoluteError()])

    # Provide training data
    X_train = np.random.random((100, 10))
    y_train = np.random.randint(2, size=(100, 1))  # Binary classification, replace with your actual task

    # Provide validation data
    X_valid = np.random.random((100, 10))
    y_valid = np.random.randint(2, size=(100, 1))  # Binary classification

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5, batch_size=32)

    # Evaluate on validation data
    _, val_acc = model.evaluate(X_valid, y_valid, verbose=0)
    print(f'Validation Accuracy: {val_acc}')

    # This is the output: validation accuracy
    return val_acc

# Call the method for validation
output = method()