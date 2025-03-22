from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: 1x1 convolution followed by 5x5 convolution
    branch3 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch3 = Conv2D(64, (5, 5), activation='relu', padding='same')(branch3)

    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(32, (1, 1), activation='relu', padding='same')(branch4)

    # Concatenate the branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the feature maps
    flattened = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(256, activation='relu')(flattened)
    dense2 = Dense(128, activation='relu')(dense1)

    # Output layer for 10 classes
    output = Dense(10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Example usage:
model = dl_model()
model.summary()