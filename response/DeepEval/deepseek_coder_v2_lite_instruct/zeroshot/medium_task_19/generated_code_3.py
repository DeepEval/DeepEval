import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First branch
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Second branch
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)

    # Third branch
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(branch3)

    # Fourth branch
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=1)(inputs)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch4)

    # Concatenate the branches
    combined = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    # Flatten the features
    flattened = Flatten()(combined)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=inputs, outputs=fc2)

    return model

# Example usage:
# model = dl_model()
# model.summary()