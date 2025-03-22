import tensorflow as tf
from tensorflow.keras import layers, models

def method():
    # Define a simple Sequential model
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(100,)),  # Example input shape
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # Example output layer with 10 classes
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # For demonstration purposes, just returning the model summary
    output = model.summary()
    return output

# Call the method for validation
method()