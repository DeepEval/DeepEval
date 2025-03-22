import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels and 1 channel (grayscale)

    # Main path
    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.Dropout(0.25)(x)

    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.25)(x)

    # Restore number of channels
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Branch path (directly connected to the input)
    branch = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Combine paths
    combined = layers.add([x, branch])

    # Flatten and fully connected layer
    combined = layers.Flatten()(combined)
    combined = layers.Dense(128, activation='relu')(combined)

    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(combined)  # 10 classes for MNIST

    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to compile and use the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))