import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Conv2D, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images shape
    num_classes = 10  # MNIST has 10 classes

    inputs = Input(shape=input_shape)

    # First Block: Processing with average pooling layers
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Flatten the pooling outputs
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    # Concatenate flattened outputs
    concatenated = Concatenate()([flat1, flat2, flat3])

    # Fully connected layer and reshape to 4D
    fc1 = Dense(128, activation='relu')(concatenated)
    reshaped = Reshape((4, 4, 8))(fc1)  # Reshape to 4D for the second block

    # Second Block: Multi-scale feature extraction
    # Path 1: 1x1 Convolution
    path1 = Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same')(reshaped)
    path1 = Dropout(0.5)(path1)

    # Path 2: 1x1 Convolution followed by two 3x3 Convolutions
    path2 = Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same')(reshaped)
    path2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
    path2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
    path2 = Dropout(0.5)(path2)

    # Path 3: 1x1 Convolution followed by 3x3 Convolution
    path3 = Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same')(reshaped)
    path3 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(path3)
    path3 = Dropout(0.5)(path3)

    # Path 4: Average Pooling followed by 1x1 Convolution
    path4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshaped)
    path4 = Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same')(path4)
    path4 = Dropout(0.5)(path4)

    # Concatenate paths outputs along the channel dimension
    concatenated_paths = Concatenate(axis=-1)([path1, path2, path3, path4])

    # Global Average Pooling to reduce spatial dimensions
    gap = GlobalAveragePooling2D()(concatenated_paths)

    # Final fully connected layers for classification
    fc2 = Dense(64, activation='relu')(gap)
    outputs = Dense(num_classes, activation='softmax')(fc2)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Instantiate the model
model = dl_model()
model.summary()