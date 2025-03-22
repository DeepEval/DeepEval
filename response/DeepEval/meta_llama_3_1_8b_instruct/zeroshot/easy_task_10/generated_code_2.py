# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.datasets import mnist

def dl_model():
    """
    Creates a deep learning model for image classification using the MNIST dataset.
    
    The model begins with a 1x1 convolutional layer to increase dimensionality,
    followed by a 3x3 depthwise separable convolutional layer for feature extraction.
    It then uses another 1x1 convolutional layer to reduce dimensionality, with a stride of 2 applied to all convolutional layers.
    Finally, the model outputs classification probabilities through a flattening layer and a fully connected layer.
    
    Returns:
        model (keras.Model): The constructed model.
    """

    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape images to have an extra dimension for convolutional layers
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # Define the model architecture
    inputs = keras.Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    x = layers.Conv2D(32, (1, 1), strides=2, activation='relu')(inputs)

    # 3x3 depthwise separable convolutional layer for feature extraction
    x = layers.DepthwiseConv2D((3, 3), strides=2, activation='relu')(x)

    # 1x1 convolutional layer to reduce dimensionality
    x = layers.Conv2D(32, (1, 1), strides=2, activation='relu')(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

    return model

# Usage
model = dl_model()
model.summary()