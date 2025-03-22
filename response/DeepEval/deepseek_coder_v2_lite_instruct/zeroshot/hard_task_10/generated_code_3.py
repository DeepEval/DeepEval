import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    path1 = Conv2D(64, (1, 1), activation='relu')(inputs)

    # Path 2: Sequence of convolutions
    path2 = Conv2D(64, (1, 1), activation='relu')(inputs)
    path2 = Conv2D(64, (1, 7), activation='relu')(path2)
    path2 = Conv2D(64, (7, 1), activation='relu')(path2)

    # Concatenate the outputs of the two paths
    concatenated = Concatenate()([path1, path2])

    # Apply a 1x1 convolution to align the output dimensions
    main_path = Conv2D(64, (1, 1), activation='relu')(concatenated)

    # Branch connection
    branch = Conv2D(64, (1, 1), activation='relu')(inputs)
    merged = Add()([main_path, branch])

    # Flatten the output
    flattened = Flatten()(merged)

    # Fully connected layers for classification
    outputs = Dense(10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()