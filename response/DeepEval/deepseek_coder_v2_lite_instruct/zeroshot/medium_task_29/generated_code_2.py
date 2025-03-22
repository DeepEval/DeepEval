import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    model = models.Sequential()

    # Add convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    # Add pooling layers
    model.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

    # Flatten the output from the pooling layers
    model.add(layers.Flatten())

    # Add fully connected layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Example usage:
# model = dl_model()
# model.summary()