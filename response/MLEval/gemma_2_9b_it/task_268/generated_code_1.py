import numpy as np
import tensorflow as tf

def method():
    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(6, activation='relu', input_shape=(input_dim,)),  # Add hidden layer
        tf.keras.layers.Dense(10, activation='softmax') 
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Example usage:
# Assume input_dim is defined elsewhere in your code
input_dim = 10 

# Call the method to create and return the model
model = method()

# You can now train and evaluate the model using model.fit() and model.evaluate()