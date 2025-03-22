import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, BatchNormalization

def dl_model():
    
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Average pooling paths
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten each path
    flattened_path1 = Flatten()(path1)
    flattened_path2 = Flatten()(path2)
    flattened_path3 = Flatten()(path3)

    # Concatenate the flattened outputs
    block1_output = Concatenate()([flattened_path1, flattened_path2, flattened_path3])

    # Fully connected layer after Block 1
    fc_layer = Dense(units=128, activation='relu')(block1_output)

    # Reshape the output to prepare for Block 2
    reshaped_output = Reshape((1, 1, 128))(fc_layer)

    # Block 2: Feature extraction branches
    # Branch 1
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Branch 2
    branch2_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch2_path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2_path1)
    branch2_path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2_path2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_path3)

    # Branch 3
    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped_output)

    # Concatenate all branches
    block2_output = Concatenate()([branch1, branch2, branch3])

    # Fully connected layers after Block 2
    flatten_block2_output = Flatten()(block2_output)
    dense1 = Dense(units=64, activation='relu')(flatten_block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model