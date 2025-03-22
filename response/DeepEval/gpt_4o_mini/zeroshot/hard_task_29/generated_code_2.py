import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for the MNIST images (28x28 pixels, 1 channel)
    input_img = layers.Input(shape=(28, 28, 1))

    # First block: Main path and branch path
    # Main path
    main_path = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
    main_path = layers.Conv2D(1, (3, 3), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_img

    # Combine paths
    combined = layers.add([main_path, branch_path])

    # Second block: Max pooling layers
    max_pool1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(combined)
    max_pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(combined)
    max_pool3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(combined)

    # Flatten the outputs of the pooling layers
    flatten1 = layers.Flatten()(max_pool1)
    flatten2 = layers.Flatten()(max_pool2)
    flatten3 = layers.Flatten()(max_pool3)

    # Concatenate the flattened outputs
    concatenated = layers.concatenate([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(concatenated)
    dense2 = layers.Dense(64, activation='relu')(dense1)

    # Output layer for classification (10 classes for digits 0-9)
    output = layers.Dense(10, activation='softmax')(dense2)

    # Create the model
    model = models.Model(inputs=input_img, outputs=output)

    return model

# Example usage
if __name__ == '__main__':
    model = dl_model()
    model.summary()

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))