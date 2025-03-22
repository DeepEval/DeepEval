from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification.
    The model consists of multiple convolutional and max pooling blocks followed
  by fully connected layers.
    
    Returns:
        A compiled Keras model for image classification.
    """

    # Input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Create the model
    model = keras.Sequential([
        # Convolutional block 1: Convolution -> Convolution -> Max Pooling
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Convolutional block 2: Convolution -> Convolution -> Convolution -> MaxPooling
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the feature maps
        layers.Flatten(),

        # Fully connected block 1: Dense -> Dropout -> Dense
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),

        # Fully connected block 2: Dense
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Load the MNIST dataset
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to have a channel dimension
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# Train the model
model = dl_model()
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))