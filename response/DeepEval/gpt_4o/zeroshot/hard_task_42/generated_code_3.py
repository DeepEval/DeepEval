from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Conv2D, AveragePooling2D, Concatenate
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))  # MNIST images are 28x28x1

    # Block 1: Three parallel max pooling paths with different scales
    # Path 1: Max Pooling with 1x1 window
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path1 = Flatten()(path1)
    path1 = Dropout(0.3)(path1)

    # Path 2: Max Pooling with 2x2 window
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path2 = Flatten()(path2)
    path2 = Dropout(0.3)(path2)

    # Path 3: Max Pooling with 4x4 window
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    path3 = Flatten()(path3)
    path3 = Dropout(0.3)(path3)

    # Concatenate results from all three paths
    concatenated = Concatenate()([path1, path2, path3])

    # Fully connected layer and reshape to 4D tensor
    fc = Dense(7 * 7 * 64, activation='relu')(concatenated)  # Assuming reshape to 7x7x64
    reshaped = Reshape((7, 7, 64))(fc)

    # Block 2: Four parallel convolutional paths
    # Path 1: Single 1x1 convolution
    path1_block2 = Conv2D(64, (1, 1), activation='relu', padding='same')(reshaped)

    # Path 2: 1x1, 1x7, and 7x1 convolutions
    path2_block2 = Conv2D(64, (1, 1), activation='relu', padding='same')(reshaped)
    path2_block2 = Conv2D(64, (1, 7), activation='relu', padding='same')(path2_block2)
    path2_block2 = Conv2D(64, (7, 1), activation='relu', padding='same')(path2_block2)

    # Path 3: 1x1 followed by alternating 7x1 and 1x7 convolutions
    path3_block2 = Conv2D(64, (1, 1), activation='relu', padding='same')(reshaped)
    path3_block2 = Conv2D(64, (7, 1), activation='relu', padding='same')(path3_block2)
    path3_block2 = Conv2D(64, (1, 7), activation='relu', padding='same')(path3_block2)
    path3_block2 = Conv2D(64, (7, 1), activation='relu', padding='same')(path3_block2)
    path3_block2 = Conv2D(64, (1, 7), activation='relu', padding='same')(path3_block2)

    # Path 4: Average Pooling followed by 1x1 convolution
    path4_block2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)
    path4_block2 = Conv2D(64, (1, 1), activation='relu', padding='same')(path4_block2)

    # Concatenate results from all four paths along the channel dimension
    concatenated_block2 = Concatenate()([path1_block2, path2_block2, path3_block2, path4_block2])

    # Final classification layers
    flattened = Flatten()(concatenated_block2)
    dense1 = Dense(256, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(dense1)  # MNIST has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model