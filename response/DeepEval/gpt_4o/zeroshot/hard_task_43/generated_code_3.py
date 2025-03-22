from tensorflow.keras import Input, Model
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1 - Three parallel paths with average pooling
    # Path 1: 1x1 pooling
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path1 = Flatten()(path1)

    # Path 2: 2x2 pooling
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path2 = Flatten()(path2)

    # Path 3: 4x4 pooling
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    path3 = Flatten()(path3)

    # Concatenate the outputs of the three paths
    block1_output = Concatenate()([path1, path2, path3])

    # Fully connected layer between Block 1 and Block 2
    fc1 = Dense(128, activation='relu')(block1_output)

    # Reshape to 4D tensor
    reshaped = Reshape((8, 8, 2))(fc1)  # Example reshape to fit the next block design

    # Block 2 - Three branches for feature extraction
    # Branch 1: <1x1 convolution, 3x3 convolution>
    branch1 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshaped)
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch1)

    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    branch2 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshaped)
    branch2 = Conv2D(16, (1, 7), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(16, (7, 1), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: Average pooling
    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)
    branch3 = Conv2D(32, (1, 1), activation='relu', padding='same')(branch3)

    # Concatenate the outputs of the three branches
    block2_output = Concatenate()([branch1, branch2, branch3])

    # Global average pooling and fully connected layers for classification
    gap = GlobalAveragePooling2D()(block2_output)
    fc2 = Dense(64, activation='relu')(gap)
    output = Dense(10, activation='softmax')(fc2)  # MNIST has 10 classes

    # Create model
    model = Model(inputs=input_layer, outputs=output)

    return model