# Import necessary packages
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def dl_model():
    """
    Function to create a deep learning model for image classification using the CIFAR-10 dataset.

    The model consists of three average pooling layers with pooling windows and strides of 1x1,2x2, and 4x4,
    allowing it to capture spatial information at different scales. The outputs of these pooling layers will be
    flattened into one-dimensional vectors and concatenated. After concatenation, the fused features will be
    further flattened and processed through two fully connected layers to generate the final classification results.
    """

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to the range [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert class labels to categorical format
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Create the deep learning model
    model = models.Sequential()

    # Convolutional layer with 32 filters, kernel size 3x3, and activation'relu'
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

    # Average pooling layer with window size 1x1 and stride 1
    model.add(layers.AveragePooling2D(pool_size=(1, 1), strides=1))

    # Convolutional layer with 64 filters, kernel size 3x3, and activation'relu'
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Average pooling layer with window size 2x2 and stride 2
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))

    # Convolutional layer with 128 filters, kernel size 3x3, and activation'relu'
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    # Average pooling layer with window size 4x4 and stride 4
    model.add(layers.AveragePooling2D(pool_size=(4, 4), strides=4))

    # Flatten the output of the convolutional and pooling layers
    model.add(layers.Flatten())

    # Concatenate the output of the first pooling layer
    model.add(layers.Concatenate([layers.Flatten()(model.layers[2].output), model.layers[4].output]))

    # Flatten the concatenated features
    model.add(layers.Flatten())

    # Dense layer with 128 units and'relu' activation
    model.add(layers.Dense(128, activation='relu'))

    # Dropout layer with dropout rate 0.2
    model.add(layers.Dropout(0.2))

    # Dense layer with 10 units and'softmax' activation (for multi-class classification)
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model