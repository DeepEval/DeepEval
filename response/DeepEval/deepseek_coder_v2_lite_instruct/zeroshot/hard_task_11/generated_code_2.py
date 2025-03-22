import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First pathway: 1x1 convolution
    pathway1 = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Second pathway: parallel branches
    branch1 = Conv2D(32, (1, 3), padding='same', activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 1), padding='same', activation='relu')(inputs)
    branch3 = Conv2D(32, (1, 3), padding='same', activation='relu')(branch1)
    branch4 = Conv2D(32, (3, 1), padding='same', activation='relu')(branch2)

    # Concatenate outputs of the two pathways
    concatenated = concatenate([pathway1, branch3, branch4])

    # Additional 1x1 convolution
    output_layer = Conv2D(32, (1, 1), activation='relu')(concatenated)

    # Flatten the output
    flattened = Flatten()(output_layer)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=inputs, outputs=fc2)

    return model

# Example usage
model = dl_model()
model.summary()