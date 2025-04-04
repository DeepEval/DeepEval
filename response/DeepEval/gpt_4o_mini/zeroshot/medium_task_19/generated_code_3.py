import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for images of shape 32x32x3 (CIFAR-10 images)
    input_layer = Input(shape=(32, 32, 3))

    # First branch: 1x1 convolution for dimensionality reduction
    branch1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_layer)

    # Second branch: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)

    # Third branch: 1x1 convolution followed by 5x5 convolution
    branch3 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(32, (5, 5), padding='same', activation='relu')(branch3)

    # Fourth branch: 3x3 max pooling followed by 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(16, (1, 1), padding='same', activation='relu')(branch4)

    # Concatenate all branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the concatenated features
    flattened = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)

    # Output layer for 10 classes (CIFAR-10)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model