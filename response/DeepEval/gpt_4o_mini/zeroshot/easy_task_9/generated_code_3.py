import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 and grayscale

    # 1x1 Convolutional Layer to increase dimensionality
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # 3x3 Depthwise Separable Convolutional Layer for feature extraction
    x = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # Another 1x1 Convolutional Layer to reduce dimensionality
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)

    # Adding the output from the last layer to the original input (skip connection)
    x = layers.add([x, inputs])

    # Flattening the output for the fully connected layer
    x = layers.Flatten()(x)

    # Fully connected layer with softmax activation for classification
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Creating the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')