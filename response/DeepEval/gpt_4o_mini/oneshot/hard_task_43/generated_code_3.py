import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, BatchNormalization

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Parallel average pooling paths
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten the outputs of the pooling layers
    flatten1 = Flatten()(path1)
    flatten2 = Flatten()(path2)
    flatten3 = Flatten()(path3)

    # Concatenate the flattened outputs
    block1_output = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layer between Block 1 and Block 2
    dense1 = Dense(units=128, activation='relu')(block1_output)

    # Reshape for Block 2 (reshaping to a suitable dimension for convolutions)
    reshape_output = Reshape((1, 1, 128))(dense1)

    # Block 2: Feature extraction branches
    # Branch 1: 1x1 convolution followed by 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_output)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_output)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: Average pooling
    branch3 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshape_output)

    # Concatenate outputs from all branches
    block2_output = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output for final classification
    flatten_block2 = Flatten()(block2_output)

    # Fully connected layers for classification
    dense2 = Dense(units=64, activation='relu')(flatten_block2)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model