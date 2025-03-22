import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Dropout, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # Define the three branches
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    branch1 = Conv2D(32, (1, 1), activation='relu')(branch1)
    branch1 = Dropout(0.5)(branch1)

    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    branch2 = Conv2D(64, (1, 1), activation='relu')(branch2)
    branch2 = Dropout(0.5)(branch2)

    branch3 = Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
    branch3 = Conv2D(128, (1, 1), activation='relu')(branch3)
    branch3 = Dropout(0.5)(branch3)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    dense1 = Dropout(0.5)(dense1)
    outputs = Dense(10, activation='softmax')(dense1)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()