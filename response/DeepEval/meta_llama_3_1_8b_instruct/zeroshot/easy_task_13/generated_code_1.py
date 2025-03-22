# Import necessary packages from Keras and other libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Normalize the input data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


def dl_model():
    # Define the model architecture
    model = keras.Sequential([
        # First 1x1 convolutional layer with dropout
        layers.Conv2D(32, (1, 1), activation='relu', input_shape=(28, 28, 1)),
        layers.Dropout(0.2),
        
        # Second 1x1 convolutional layer with dropout
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Dropout(0.2),
        
        # 3x1 convolutional layer with dropout
        layers.Conv2D(32, (3, 1), activation='relu'),
        layers.Dropout(0.2),
        
        # 1x3 convolutional layer with dropout
        layers.Conv2D(32, (1, 3), activation='relu'),
        layers.Dropout(0.2),
        
        # Restore the number of channels to match the input's channel count
        layers.Conv2D(1, (1, 1), activation='relu'),
        
        # Add the processed features with the original input
        layers.Add()([layers.Conv2D(1, (1, 1), activation='relu')(layers.Concatenate()([inputs, layers.Conv2D(1, (1, 1), activation='relu')(inputs)]))]),
        
        # Flatten the output
        layers.Flatten(),
        
        # Fully connected layer to produce the final probability distribution
        layers.Dense(10, activation='softmax')
    ])
    
    return model


# Define the input layer
inputs = keras.Input(shape=(28, 28, 1))

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))