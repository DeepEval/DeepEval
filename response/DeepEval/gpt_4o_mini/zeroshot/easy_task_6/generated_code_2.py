import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(28, 28, 1))

    # Main path
    # First convolutional layer that increases the feature map width
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    # Second convolutional layer that restores the number of channels
    x = layers.Conv2D(1, (3, 3), padding='same', activation='relu')(x)

    # Branch path directly connecting to the input
    branch = layers.Conv2D(1, (1, 1), padding='same')(input_layer)

    # Combine the paths through an addition operation
    combined = layers.add([x, branch])

    # Flatten layer
    flattened = layers.Flatten()(combined)

    # Fully connected layer
    dense = layers.Dense(128, activation='relu')(flattened)

    # Output layer with softmax activation for classification
    output_layer = layers.Dense(10, activation='softmax')(dense)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Optionally, if you want to compile and summarize the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')