import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    block1_output = x

    # Second block
    x = Conv2D(64, (3, 3), padding='same')(block1_output)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    block2_output = x

    # Concatenate the outputs of the first and second blocks
    x = Concatenate(axis=-1)([block1_output, block2_output])

    # Third block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    block3_output = x

    # Flatten the output
    x = Flatten()(block3_output)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model