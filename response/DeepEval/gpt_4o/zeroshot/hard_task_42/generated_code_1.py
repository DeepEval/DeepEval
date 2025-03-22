from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Concatenate, Reshape, Conv2D, AveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for MNIST images
    input_layer = Input(shape=(28, 28, 1))

    # Block 1 - Parallel Max Pooling Paths
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path1 = Flatten()(path1)
    path1 = Dropout(0.5)(path1)

    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path2 = Flatten()(path2)
    path2 = Dropout(0.5)(path2)

    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    path3 = Flatten()(path3)
    path3 = Dropout(0.5)(path3)

    # Concatenate the outputs of all three paths
    concatenated_block1 = Concatenate()([path1, path2, path3])

    # Fully connected layer between blocks 1 and 2
    fc_between = Dense(1024, activation='relu')(concatenated_block1)

    # Reshape the output to a 4D tensor for block 2 processing
    reshape_layer = Reshape((4, 4, 64))(fc_between)

    # Block 2 - Parallel Convolution Paths
    # Path 1: Single 1x1 Convolution
    path1_block2 = Conv2D(64, (1, 1), activation='relu', padding='same')(reshape_layer)

    # Path 2: 1x1 Convolution followed by 1x7 and 7x1 Convolutions
    path2_block2 = Conv2D(64, (1, 1), activation='relu', padding='same')(reshape_layer)
    path2_block2 = Conv2D(64, (1, 7), activation='relu', padding='same')(path2_block2)
    path2_block2 = Conv2D(64, (7, 1), activation='relu', padding='same')(path2_block2)

    # Path 3: 1x1 Convolution followed by alternating 7x1 and 1x7 Convolutions
    path3_block2 = Conv2D(64, (1, 1), activation='relu', padding='same')(reshape_layer)
    path3_block2 = Conv2D(64, (7, 1), activation='relu', padding='same')(path3_block2)
    path3_block2 = Conv2D(64, (1, 7), activation='relu', padding='same')(path3_block2)
    path3_block2 = Conv2D(64, (7, 1), activation='relu', padding='same')(path3_block2)
    path3_block2 = Conv2D(64, (1, 7), activation='relu', padding='same')(path3_block2)

    # Path 4: Average Pooling followed by 1x1 Convolution
    path4_block2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshape_layer)
    path4_block2 = Conv2D(64, (1, 1), activation='relu', padding='same')(path4_block2)

    # Concatenate the outputs of all paths in block 2
    concatenated_block2 = Concatenate()([path1_block2, path2_block2, path3_block2, path4_block2])
    flattened_block2 = Flatten()(concatenated_block2)

    # Final fully connected layers for classification
    fc_final1 = Dense(512, activation='relu')(flattened_block2)
    output_layer = Dense(10, activation='softmax')(fc_final1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Usage
model = dl_model()
model.summary()