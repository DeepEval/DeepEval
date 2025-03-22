import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer with shape (32, 32, 64)
    input_shape = (32, 32, 64)
    inputs = Input(shape=input_shape)

    # Compressing the input channels using a 1x1 convolutional layer
    compressed = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Expanding features through two parallel convolutional layers
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(compressed)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(compressed)

    # Concatenating the results of the two convolutions
    concatenated = Concatenate()([conv1, conv2])

    # Flattening the output feature map into a one-dimensional vector
    flattened = Flatten()(concatenated)

    # Fully connected layer 1
    dense1 = Dense(units=128, activation='relu')(flattened)

    # Fully connected layer 2
    dense2 = Dense(units=10, activation='softmax')(dense1)  # Assuming 10 classes for output

    # Constructing the model
    model = Model(inputs=inputs, outputs=dense2)

    return model

# Example usage
model = dl_model()
model.summary()