from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Dropout, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Normalize the input image data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(input_shape)

    # First convolutional and max pooling layer
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2))(x)

    # Second convolutional and max pooling layer
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2))(x)

    # Upsampling and convolutional layer with skip connection
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = concatenate([x, Conv2D(64, (3, 3), padding='same')(x)])
    x = LeakyReLU(alpha=0.2)(x)

    # Upsampling and convolutional layer with skip connection
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = concatenate([x, Conv2D(32, (3, 3), padding='same')(x)])
    x = LeakyReLU(alpha=0.2)(x)

    # Additional convolutional and dropout layer
    x = Conv2D(10, (3, 3), padding='same')(x)
    x = Dropout(0.2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # 1x1 convolutional layer for probability output
    outputs = Conv2D(10, (1, 1), activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model