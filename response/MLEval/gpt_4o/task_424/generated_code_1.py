import tensorflow as tf
from tensorflow.keras import layers, models

def method():
    # Define a simple Sequential model
    model = models.Sequential()
    
    # Add layers to the model
    model.add(layers.Dense(64, activation='relu', input_shape=(32,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Summary of the model
    model.summary()
    
    # Return the model as the output, which is now reset to its initial state
    return model

# Call the method to validate it initializes the model
model = method()