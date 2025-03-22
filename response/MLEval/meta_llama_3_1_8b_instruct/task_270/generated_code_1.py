# Import necessary packages
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def method():
    """
    This function sets up an actual MLP model using Keras.
    """
    
    # Generate some dummy data for testing
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    
    # Set up the MLP model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, epochs=10, verbose=0)
    
    # Return the trained model
    return model

# Call the generated method for validation
output = method()

# Print the summary of the model
output.summary()