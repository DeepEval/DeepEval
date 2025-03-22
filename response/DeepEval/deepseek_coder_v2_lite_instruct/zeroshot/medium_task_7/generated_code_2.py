import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First convolutional layer
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)

    # Second convolutional layer
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)

    # Third convolutional layer
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)

    # Add the output of the first two convolutions to the output of the third convolution
    added = Add()([x1, x2, x3])

    # Flatten the output
    flattened = Flatten()(added)

    # First fully connected layer
    dense1 = Dense(128, activation='relu')(flattened)

    # Second fully connected layer (output layer for classification)
    outputs = Dense(10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()