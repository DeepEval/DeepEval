from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First branch: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Second branch: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)

    # Third branch: 1x1 convolution followed by 5x5 convolution
    branch3 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(32, (5, 5), padding='same', activation='relu')(branch3)

    # Fourth branch: 3x3 max pooling followed by 1x1 convolution
    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenate all branches
    concatenated = concatenate([branch1, branch2, branch3, branch4], axis=-1)

    # Flatten the concatenated features
    flat = Flatten()(concatenated)

    # Fully connected layers
    fc1 = Dense(256, activation='relu')(flat)
    fc2 = Dense(128, activation='relu')(fc1)

    # Output layer for 10 classes
    output_layer = Dense(10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model