import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel (grayscale)
    inputs = Input(shape=input_shape)

    # Main path
    x = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = SeparableConv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch path
    branch = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)

    # Match dimensions using a 1x1 convolution
    branch = Conv2D(64, (1, 1), padding='same')(branch)

    # Sum the outputs of both paths
    x = Add()([x, branch])

    # Flattening and output layer
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of using the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)  # Adding channel dimension
x_test = np.expand_dims(x_test, axis=-1)  # Adding channel dimension
x_train = x_train.astype('float32') / 255.0  # Normalize to [0, 1]
x_test = x_test.astype('float32') / 255.0  # Normalize to [0, 1]
y_train = to_categorical(y_train, num_classes=10)  # One-hot encoding
y_test = to_categorical(y_test, num_classes=10)  # One-hot encoding

# Fit the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))