import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First convolutional layer
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = MaxPooling2D((2, 2))(x1)

    # Second convolutional layer
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x2 = MaxPooling2D((2, 2))(x2)

    # Third convolutional layer
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x3 = MaxPooling2D((2, 2))(x3)

    # Direct processing of input
    x_direct = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x_direct = MaxPooling2D((2, 2))(x_direct)

    # Adding outputs from different paths
    added_outputs = Add()([x1, x2, x3, x_direct])

    # Flatten the output
    flattened = Flatten()(added_outputs)

    # First fully connected layer
    fc1 = Dense(128, activation='relu')(flattened)

    # Second fully connected layer (output layer)
    outputs = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()