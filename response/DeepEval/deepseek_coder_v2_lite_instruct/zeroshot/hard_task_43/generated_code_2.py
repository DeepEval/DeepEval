import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate, Reshape

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Path 1: 1x1 average pooling
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(path1)

    # Path 2: 2x2 average pooling
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(path2)

    # Path 3: 4x4 average pooling
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(path3)

    # Concatenate the outputs of the three paths
    concat1 = Concatenate(axis=-1)([Flatten()(path1), Flatten()(path2), Flatten()(path3)])

    # Fully connected layer after Block 1
    fc1 = Dense(128, activation='relu')(concat1)

    # Reshape the output to 4-dimensional tensor
    reshape1 = Reshape((1, 1, 128))(fc1)

    # Block 2
    # Branch 1: 1x1 convolution, 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(reshape1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(reshape1)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same')(branch2)

    # Branch 3: 3x3 average pooling
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(reshape1)
    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(branch3)

    # Concatenate the outputs of the three branches
    concat2 = Concatenate(axis=-1)([Flatten()(branch1), Flatten()(branch2), Flatten()(branch3)])

    # Fully connected layers after Block 2
    fc2 = Dense(64, activation='relu')(concat2)
    output_layer = Dense(10, activation='softmax')(fc2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model