import tensorflow as tf
from tensorflow import keras

def method():
    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(784,)),  # Input layer
        keras.layers.Dense(32, activation='relu'),  # Hidden layer
        keras.layers.Dense(10, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Call the method and store the output
model = method()

# Print the model summary
model.summary()