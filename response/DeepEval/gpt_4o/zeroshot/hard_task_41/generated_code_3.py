import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dropout, Dense, Reshape, Conv2D, concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 with a single channel

    # Input layer
    inputs = Input(shape=input_shape)

    # Block 1
    # Path 1: 1x1 Average Pooling
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    path1 = Flatten()(path1)
    path1 = Dropout(0.5)(path1)

    # Path 2: 2x2 Average Pooling
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    path2 = Flatten()(path2)
    path2 = Dropout(0.5)(path2)

    # Path 3: 4x4 Average Pooling
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)
    path3 = Flatten()(path3)
    path3 = Dropout(0.5)(path3)

    # Concatenate paths
    concatenated = concatenate([path1, path2, path3])

    # Fully connected layer and reshape for block 2
    fc = Dense(128, activation='relu')(concatenated)
    reshaped = Reshape((4, 4, 8))(fc)  # Adjust dimensions accordingly

    # Block 2
    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(8, (1, 1), activation='relu', padding='same')(reshaped)

    # Branch 2: 1x1 Convolution -> 3x3 Convolution
    branch2 = Conv2D(8, (1, 1), activation='relu', padding='same')(reshaped)
    branch2 = Conv2D(8, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: 1x1 Convolution -> 3x3 Convolution -> 3x3 Convolution
    branch3 = Conv2D(8, (1, 1), activation='relu', padding='same')(reshaped)
    branch3 = Conv2D(8, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(8, (3, 3), activation='relu', padding='same')(branch3)

    # Branch 4: Average Pooling -> 1x1 Convolution
    branch4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshaped)
    branch4 = Conv2D(8, (1, 1), activation='relu', padding='same')(branch4)

    # Concatenate branches
    concatenated_branches = concatenate([branch1, branch2, branch3, branch4])

    # Flatten and final fully connected layers
    flat = Flatten()(concatenated_branches)
    fc1 = Dense(64, activation='relu')(flat)
    output = Dense(10, activation='softmax')(fc1)  # 10 classes for MNIST

    # Model
    model = Model(inputs=inputs, outputs=output)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()