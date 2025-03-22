import keras
from keras.models import Sequential
from keras.layers import Dense

def method():
    # Create a sequential model
    model = Sequential()

    # Add an input layer
    model.add(Dense(64, activation='relu', input_shape=(784,)))

    # Add a hidden layer
    model.add(Dense(6, activation='relu'))

    # Add another hidden layer
    model.add(Dense(6, activation='relu'))

    # Add an output layer
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the model
    return model

# Call the method for validation
model = method()