# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape input data
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

def dl_model():
    """
    Create a deep learning model for image classification using the MNIST dataset.
    
    The model features three branches, each built from a specialized block. 
    This block begins with a depthwise separable convolutional layer, 
    followed by a 1x1 convolutional layer to extract features. 
    To mitigate overfitting, dropout layers are applied after both convolutional layers. 
    The outputs from the three branches are concatenated and then processed through two fully connected layers 
    to generate the final classification results.
    
    Returns:
    A compiled Keras model.
    """

    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the model
    model = keras.Sequential()

    # Branch 1
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())

    # Branch 2
    model.add(layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())

    # Branch 3
    model.add(layers.DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())

    # Concatenate outputs from three branches
    model.add(layers.Concatenate()([model.layers[-6].output, model.layers[-5].output, model.layers[-4].output]))

    # Two fully connected layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Test the function
if __name__ == "__main__":
    model = dl_model()
    model.summary()