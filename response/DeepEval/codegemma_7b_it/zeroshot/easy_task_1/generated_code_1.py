import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Construct the model using Functional API
    inputs = keras.Input(shape=(28, 28, 1))  # Define the input layer

    # Add convolutional layers with max pooling
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the feature maps
    x = layers.Flatten()(x)

    # Add fully connected layers for classification
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model