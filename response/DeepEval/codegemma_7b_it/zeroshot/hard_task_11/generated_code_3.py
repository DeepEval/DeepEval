import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main pathway
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    # Parallel branches
    x_1x1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x_1x3 = layers.Conv2D(64, (1, 3), padding='same', activation='relu')(x)
    x_3x1 = layers.Conv2D(64, (3, 1), padding='same', activation='relu')(x)

    # Concatenation and 1x1 convolution
    concat = layers.concatenate([x_1x1, x_1x3, x_3x1])
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(concat)

    # Direct connection
    shortcut = layers.Conv2D(64, (1, 1), padding='same')(inputs)

    # Fusion
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    # Classification layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Model creation
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model