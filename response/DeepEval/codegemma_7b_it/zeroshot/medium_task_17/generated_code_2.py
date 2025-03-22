import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Permute, Dense, Activation

def dl_model():

    # Input layer
    input_shape = (None, None, 3)
    inputs = Input(shape=input_shape)

    # Reshape and reshape with groups
    x = Reshape((input_shape[0], input_shape[1], 3, 1))(inputs)
    x = Reshape((input_shape[0], input_shape[1], 1, 3))(x)

    # Permute to enable channel shuffling
    x = Permute((0, 1, 3, 2))(x)

    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)

    # Model construction
    model = Model(inputs=inputs, outputs=outputs)

    return model