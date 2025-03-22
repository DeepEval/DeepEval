import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 2: 1x1 convolution followed by a 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolution to adjust the output dimensions
    adjusted = Conv2D(64, (1, 1), activation='relu')(concatenated)

    # Add the main path and the branch
    fused = Add()([inputs, adjusted])

    # Flatten the output
    flattened = Flatten()(fused)

    # Three fully connected layers for classification
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(64, activation='relu')(fc1)
    outputs = Dense(10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()