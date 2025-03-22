import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)

    # Max pooling layer
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Adding the input layer to the output of the convolutional layers
    x = layers.add([x, inputs])

    # Flattening the features
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Constructing the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# If you'd like to see the model summary, you can uncomment the lines below
# model = dl_model()
# model.summary()