import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Dropout, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch1 = Dropout(0.25)(branch1)
    branch1 = Conv2D(64, (1, 1), activation='relu')(branch1)
    branch1 = Dropout(0.25)(branch1)

    # Branch 2
    branch2 = DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch2 = Dropout(0.25)(branch2)
    branch2 = Conv2D(64, (1, 1), activation='relu')(branch2)
    branch2 = Dropout(0.25)(branch2)

    # Branch 3
    branch3 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch3 = Dropout(0.25)(branch3)
    branch3 = Conv2D(64, (1, 1), activation='relu')(branch3)
    branch3 = Dropout(0.25)(branch3)

    # Concatenate the outputs from the three branches
    combined = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = Flatten()(combined)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=input_layer, outputs=fc2)

    return model

# Example usage
model = dl_model()
model.summary()