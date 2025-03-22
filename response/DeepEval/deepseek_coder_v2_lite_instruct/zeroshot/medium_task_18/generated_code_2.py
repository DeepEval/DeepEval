import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    model = models.Sequential()

    # Block 1: 3x3, 64 filters, ReLU activation
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Block 2: 3x3, 128 filters, ReLU activation
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Block 3: 3x3, 256 filters, ReLU activation
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Block 4: 1x1, 512 filters, ReLU activation
    model.add(layers.Conv2D(512, (1, 1), activation='relu'))
    model.add(layers.BatchNormalization())

    # Block 5: 3x3, 512 filters, ReLU activation
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Block 6: 5x5, 512 filters, ReLU activation
    model.add(layers.Conv2D(512, (5, 5), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Flatten layer
    model.add(layers.Flatten())

    # Fully connected layer 1: 256 units, ReLU activation
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())

    # Dropout layer
    model.add(layers.Dropout(0.5))

    # Fully connected layer 2: 128 units, ReLU activation
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())

    # Dropout layer
    model.add(layers.Dropout(0.5))

    # Output layer: 10 units (CIFAR-10 classes), softmax activation
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Example usage
# model = dl_model()
# model.summary()