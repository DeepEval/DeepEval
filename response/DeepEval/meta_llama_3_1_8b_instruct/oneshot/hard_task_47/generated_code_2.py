import keras
from keras.layers import Input, Concatenate, Lambda, BatchNormalization, DepthwiseConv2D, Reshape, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, concatenate
from keras import backend as K
from keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import numpy as np

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_input)(input_layer)

    # Define three branches for depthwise separable convolution
    branch1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    branch1 = BatchNormalization()(branch1)
    branch2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    branch2 = BatchNormalization()(branch2)
    branch3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    branch3 = BatchNormalization()(branch3)

    # Concatenate the outputs from the three branches
    concatenated_layer = Concatenate()([branch1, branch2, branch3])

    # Define the second block
    # First branch: 1x1 convolution + 3x3 convolution
    branch4 = Conv2D(kernel_size=(1, 1), padding='same')(concatenated_layer)
    branch4 = BatchNormalization()(branch4)
    branch4 = Activation('relu')(branch4)
    branch4 = Conv2D(kernel_size=(3, 3), padding='same')(branch4)
    branch4 = BatchNormalization()(branch4)
    branch4 = Activation('relu')(branch4)

    # Second branch: 1x1 convolution + 1x7 convolution + 7x1 convolution + 3x3 convolution
    branch5 = Conv2D(kernel_size=(1, 1), padding='same')(concatenated_layer)
    branch5 = BatchNormalization()(branch5)
    branch5 = Activation('relu')(branch5)
    branch5 = Conv2D(kernel_size=(1, 7), padding='same')(branch5)
    branch5 = BatchNormalization()(branch5)
    branch5 = Activation('relu')(branch5)
    branch5 = Conv2D(kernel_size=(7, 1), padding='same')(branch5)
    branch5 = BatchNormalization()(branch5)
    branch5 = Activation('relu')(branch5)
    branch5 = Conv2D(kernel_size=(3, 3), padding='same')(branch5)
    branch5 = BatchNormalization()(branch5)
    branch5 = Activation('relu')(branch5)

    # Third branch: average pooling
    branch6 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(concatenated_layer)

    # Concatenate the outputs from all branches
    concatenated_layer = Concatenate()([branch4, branch5, branch6])

    # Flatten the output
    flatten_layer = Flatten()(concatenated_layer)

    # Define the two fully connected layers
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model