import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Add

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # First branch: 3x3 convolutions
    branch1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = MaxPooling2D((2, 2))(branch1)

    # Second branch: 1x1 convolutions followed by two 3x3 convolutions
    branch2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D((2, 2))(branch2)

    # Third branch: max pooling
    branch3 = MaxPooling2D((2, 2))(inputs)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Add two fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()