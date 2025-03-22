import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 color channel

    # 1x1 Convolutional Layer to increase dimensionality
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(inputs)

    # 3x3 Depthwise Separable Convolutional Layer
    x = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)

    # Another 1x1 Convolutional Layer to reduce dimensionality
    x = layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    outputs = layers.Dense(units=10, activation='softmax')(x)  # 10 classes for MNIST digits 0-9

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of how to load the MNIST dataset and preprocess it (not part of the dl_model function)
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0  # Reshape and normalize
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return (x_train, y_train), (x_test, y_test)

# Usage
model = dl_model()
model.summary()  # Display the model architecture