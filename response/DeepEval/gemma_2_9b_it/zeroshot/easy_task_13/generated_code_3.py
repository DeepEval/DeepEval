import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Input layer for MNIST images (28x28 grayscale)
        
        layers.Conv1D(filters=32, kernel_size=1, activation='relu', padding='same'),
        layers.Dropout(0.25), 

        layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same'),
        layers.Dropout(0.25), 

        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.Dropout(0.25), 

        layers.Conv1D(filters=32, kernel_size=1, activation='relu', padding='same'),
        layers.Dropout(0.25), 

        layers.Conv1D(filters=1, kernel_size=1, activation='sigmoid', padding='same'),  # Restore original channel count
        
        layers.Add()  # Add the processed features to the original input

        layers.Flatten(), 
        layers.Dense(10, activation='softmax')  # Output layer for 10 MNIST classes
    ])

    return model