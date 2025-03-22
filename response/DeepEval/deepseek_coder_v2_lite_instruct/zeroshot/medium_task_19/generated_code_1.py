import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input shape
    inputs = Input(shape=(32, 32, 3))

    # First branch
    branch1 = Conv2D(64, (1, 1), activation='relu')(inputs)

    # Second branch
    branch2 = Conv2D(64, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)

    # Third branch
    branch3 = Conv2D(64, (1, 1), activation='relu')(inputs)
    branch3 = Conv2D(64, (5, 5), activation='relu', padding='same')(branch3)

    # Fourth branch
    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch4 = Conv2D(64, (1, 1), activation='relu')(branch4)

    # Concatenate the branches
    combined = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    # Flatten the features
    flattened = Flatten()(combined)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=inputs, outputs=fc2)

    return model

# Example usage:
# model = dl_model()
# model.summary()