import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Conv2D, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Parallel Average Pooling paths
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten the outputs of the pooling layers
    flat_path1 = Flatten()(path1)
    flat_path2 = Flatten()(path2)
    flat_path3 = Flatten()(path3)

    # Concatenate the flattened outputs
    block1_output = Concatenate()([flat_path1, flat_path2, flat_path3])

    # Fully connected layer after Block 1
    dense1 = Dense(units=128, activation='relu')(block1_output)

    # Reshape to prepare for Block 2
    reshape_layer = Reshape((1, 1, 128))(dense1)  # Reshape to 4D tensor (1, 1, 128) for Block 2 processing

    # Block 2: Feature extraction with three branches
    # Branch 1: <1x1 Convolution, 3x3 Convolution>
    branch1_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch1_conv1)

    # Branch 2: <1x1 Convolution, 1x7 Convolution, 7x1 Convolution, 3x3 Convolution>
    branch2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch2_conv2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(branch2_conv1)
    branch2_conv3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(branch2_conv2)
    branch2_conv4 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2_conv3)

    # Branch 3: Average Pooling
    branch3 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshape_layer)

    # Concatenate outputs from all branches
    block2_output = Concatenate()([branch1_conv2, branch2_conv4, branch3])

    # Flatten the concatenated output from Block 2
    flat_block2_output = Flatten()(block2_output)

    # Fully connected layers for final classification
    dense2 = Dense(units=64, activation='relu')(flat_block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model