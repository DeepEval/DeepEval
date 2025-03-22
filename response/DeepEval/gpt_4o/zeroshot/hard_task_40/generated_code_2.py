import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Concatenate, Reshape, Conv2D, Dropout
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)

    # First block
    # Three average pooling layers with different window sizes and strides
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Flattening the output of each pooling layer
    flat_1x1 = Flatten()(avg_pool_1x1)
    flat_2x2 = Flatten()(avg_pool_2x2)
    flat_4x4 = Flatten()(avg_pool_4x4)

    # Concatenating the flattened vectors
    concatenated = Concatenate()([flat_1x1, flat_2x2, flat_4x4])

    # Fully connected layer
    fc1 = Dense(128, activation='relu')(concatenated)

    # Reshape to 4D tensor for the second block
    reshape = Reshape((4, 4, 8))(fc1)  # Example reshape dimensions

    # Second block with four paths
    # Path 1: 1x1 convolution
    path1 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshape)
    path1 = Dropout(0.3)(path1)

    # Path 2: 1x1 convolution followed by two 3x3 convolutions
    path2 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshape)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)
    path2 = Dropout(0.3)(path2)

    # Path 3: 1x1 convolution followed by a 3x3 convolution
    path3 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshape)
    path3 = Conv2D(32, (3, 3), activation='relu', padding='same')(path3)
    path3 = Dropout(0.3)(path3)

    # Path 4: Average pooling followed by a 1x1 convolution
    path4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshape)
    path4 = Conv2D(16, (1, 1), activation='relu', padding='same')(path4)
    path4 = Dropout(0.3)(path4)

    # Concatenating outputs from all paths along the channel dimension
    concatenated_paths = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flat_paths = Flatten()(concatenated_paths)

    # Fully connected layers for classification
    fc2 = Dense(128, activation='relu')(flat_paths)
    outputs = Dense(num_classes, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model