import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    x1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (3, 3), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (3, 3), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (3, 3), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (3, 3), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)

    # Block 2
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)

    # Concatenate the outputs of the two blocks
    x = Concatenate(axis=-1)([x1, x2])

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()