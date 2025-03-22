from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x = layers.Conv2D(filters=32, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    branch = keras.Input(shape=(28, 28, 1))
    branch_output = layers.Conv2D(filters=64, kernel_size=3, padding='same')(branch)
    branch_output = layers.BatchNormalization()(branch_output)
    branch_output = layers.Activation('relu')(branch_output)

    outputs = layers.add([x, branch_output])
    outputs = layers.Activation('relu')(outputs)

    # Block 2
    x = layers.MaxPooling2D(pool_size=1, strides=1)(outputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=4, strides=4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Flatten()(x)

    # Block 3
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(10)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('softmax')(x)

    model = keras.Model(inputs=[inputs, branch], outputs=x)

    return model