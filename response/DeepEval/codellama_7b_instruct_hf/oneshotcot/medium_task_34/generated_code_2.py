import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.cifar10 import Cifar10



def dl_model():

    # Load the CIFAR-10 dataset
    cifar10 = Cifar10()

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first part of the model, which consists of 3 pairs of convolutional layers and max pooling layers
    conv_layers = [
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_shape),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layers[0]),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_layers[1])
    ]

    # Define the second part of the model, which consists of a convolutional layer, a dropout layer, and another convolutional layer
    conv_layers_2 = [
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layers[2]),
        Dropout(rate=0.5)(conv_layers[2]),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layers[2])
    ]

    # Define the third part of the model, which consists of 3 pairs of convolutional layers and transposed convolutional layers
    conv_layers_3 = [
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layers_2),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layers_3),
        TransposedConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layers_3),
        TransposedConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layers_3),
        TransposedConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layers_3)
    ]

    # Define the final convolutional layer
    conv_layers_4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layers_3)

    # Define the final flatten layer
    flatten_layer = Flatten()(conv_layers_4)

    # Define the final dense layers
    dense_layers = [
        Dense(units=10, activation='softmax')(flatten_layer),
        Dense(units=10, activation='softmax')(dense_layers[0]),
        Dense(units=10, activation='softmax')(dense_layers[1])
    ]

    # Define the model
    model = Model(inputs=input_shape, outputs=dense_layers[2])

    return model