import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First branch
    branch1 = Conv2D(64, (1, 1), activation='relu')(inputs)

    # Second branch
    branch2 = Conv2D(64, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)

    # Third branch
    branch3 = Conv2D(64, (1, 1), activation='relu')(inputs)
    branch3 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)

    # Concatenate branches
    merged = Concatenate()([branch1, branch2, branch3])

    # Adjust dimensions
    merged = Conv2D(64, (1, 1), activation='relu')(merged)

    # Flatten the output
    flattened = Flatten()(merged)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(64, activation='relu')(fc1)
    outputs = Dense(10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model