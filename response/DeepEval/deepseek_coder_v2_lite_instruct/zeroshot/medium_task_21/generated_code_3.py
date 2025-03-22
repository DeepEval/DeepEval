import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch4 = Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs from all branches
    concatenated = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Add dropout layers
    dropout1 = Dropout(0.5)(flattened)
    dropout2 = Dropout(0.5)(dropout1)

    # Fully connected layers
    outputs = Dense(10, activation='softmax')(dropout2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()