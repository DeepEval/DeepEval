import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model

def dl_model():
    # Define input shape for MNIST dataset (28x28 pixels with 1 channel)
    input_shape = (28, 28, 1)
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)

    # First convolutional block
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)

    # Second convolutional block
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x2 = MaxPooling2D((2, 2), padding='same')(x2)

    # Ensure last Conv2D layer output dimension matches the input channel
    x3 = Conv2D(1, (3, 3), activation='relu', padding='same')(x2)

    # Adding the last convolutional output to the original input
    added = Add()([x3, inputs])

    # Flattening the output and adding a fully connected layer
    flat = Flatten()(added)
    outputs = Dense(num_classes, activation='softmax')(flat)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()