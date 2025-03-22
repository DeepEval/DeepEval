import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input shape
    inputs = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    branch1 = Conv2D(32, (3, 3), activation='relu')(inputs)

    # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
    branch2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)

    # Branch 3: Max pooling followed by two 3x3 convolutions
    branch3 = MaxPooling2D((3, 3), strides=(1, 1))(inputs)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)

    # Concatenate the outputs of the three branches
    merged = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated feature maps
    flattened = Flatten()(merged)

    # Add two fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()