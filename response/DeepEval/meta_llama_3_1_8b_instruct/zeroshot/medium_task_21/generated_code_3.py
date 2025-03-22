# Import necessary packages
from keras import layers
from keras import models
from keras import regularizers
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize input data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Define branch 1: 1x1 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch1 = layers.Dropout(0.2)(branch1)

    # Define branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = layers.Dropout(0.2)(branch2)

    # Define branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = layers.Dropout(0.2)(branch3)

    # Define branch 4: Average pooling followed by 1x1 convolution
    branch4 = layers.AveragePooling2D((2, 2))(inputs)
    branch4 = layers.Conv2D(32, (1, 1), activation='relu')(branch4)
    branch4 = layers.Dropout(0.2)(branch4)

    # Concatenate the outputs from all branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Define the convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu')(concatenated)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Define three fully connected layers for classification
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = models.Model(inputs=inputs, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model