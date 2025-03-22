import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images have a size of 32x32 with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # First convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    # Second convolutional layer
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    
    # Max pooling layer
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Adding the input layer to the output features from the convolutional layers
    x = layers.add([x, inputs])  # Skip connection

    # Flattening the features
    x = layers.Flatten()(x)

    # First fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    # Second fully connected layer
    x = layers.Dense(64, activation='relu')(x)

    # Output layer with softmax activation for 10 classes
    outputs = layers.Dense(10, activation='softmax')(x)

    # Creating the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to use the model function
model = dl_model()
model.summary()