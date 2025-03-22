import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def method():
    # Create a sequential model
    model = Sequential()

    # Add a dense layer with 64 units and a ReLU activation function
    model.add(Dense(64, activation='relu', input_shape=(784,)))

    # Add a dropout layer with a dropout rate of 0.2
    model.add(Dropout(0.2))

    # Add a dense layer with 10 units and a softmax activation function
    model.add(Dense(10, activation='softmax'))

    # Compile the model with an Adam optimizer and a categorical cross-entropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the compiled model
    return model

# Call the generated 'method()' for validation
model = method()


# Train the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the validation data
loss, accuracy = model.evaluate(X_val, y_val)

print(f'Loss: {loss:.3f}, Accuracy: {accuracy:.3f}')