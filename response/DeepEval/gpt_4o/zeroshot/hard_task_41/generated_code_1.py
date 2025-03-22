from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Dense, Reshape, Concatenate, Conv2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Parallel Average Pooling Paths
    pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    flat_1x1 = Flatten()(pool_1x1)
    drop_1x1 = Dropout(0.5)(flat_1x1)
    
    pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    flat_2x2 = Flatten()(pool_2x2)
    drop_2x2 = Dropout(0.5)(flat_2x2)
    
    pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    flat_4x4 = Flatten()(pool_4x4)
    drop_4x4 = Dropout(0.5)(flat_4x4)

    # Concatenate the results from the pooling paths
    concatenated = Concatenate()([drop_1x1, drop_2x2, drop_4x4])

    # Fully connected layer and Reshape for Block 2
    fc = Dense(7 * 7 * 16, activation='relu')(concatenated)
    reshaped = Reshape((7, 7, 16))(fc)

    # Block 2: Multiple Branch Connections
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshaped)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshaped)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: 1x1 convolution, 3x3 convolution, 3x3 convolution
    branch3 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshaped)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)
    branch4 = Conv2D(16, (1, 1), activation='relu', padding='same')(branch4)

    # Concatenate the outputs of the branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3, branch4])

    # Final classification layers
    flat_final = Flatten()(concatenated_branches)
    dense1 = Dense(128, activation='relu')(flat_final)
    output_layer = Dense(10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model