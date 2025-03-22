import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_img = layers.Input(shape=(28, 28, 1))

    # Main path
    # First convolutional layer that increases the feature map width
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
    # Second convolutional layer that restores the number of channels
    x = layers.Conv2D(1, (3, 3), padding='same', activation='relu')(x)

    # Branch path
    # Directly connect to the input
    branch = input_img

    # Combine paths using addition
    combined = layers.add([x, branch])

    # Flatten the combined output
    x = layers.Flatten()(combined)

    # Fully connected layer for classification
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=input_img, outputs=output)

    return model

# Example of how to use the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))