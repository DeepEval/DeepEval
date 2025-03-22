import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels, 3 color channels)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Average Pooling layer with 1x1 pool size
    pool1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)

    # Average Pooling layer with 2x2 pool size
    pool2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)

    # Average Pooling layer with 4x4 pool size
    pool3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flattening the output of each pooling layer
    flat1 = layers.Flatten()(pool1)
    flat2 = layers.Flatten()(pool2)
    flat3 = layers.Flatten()(pool3)

    # Concatenating the flattened outputs
    concatenated = layers.concatenate([flat1, flat2, flat3])

    # Fully connected layer 1
    dense1 = layers.Dense(512, activation='relu')(concatenated)

    # Fully connected layer 2
    dense2 = layers.Dense(256, activation='relu')(dense1)

    # Output layer with 10 units for CIFAR-10 classes and softmax activation
    output_layer = layers.Dense(10, activation='softmax')(dense2)

    # Creating the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to compile the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Fit the model (example)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)