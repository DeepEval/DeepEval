from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x_1x1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x_2x2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    x_4x4 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    x_1x1 = layers.Flatten()(x_1x1)
    x_1x1 = layers.Dropout(0.2)(x_1x1)

    x_2x2 = layers.Flatten()(x_2x2)
    x_2x2 = layers.Dropout(0.2)(x_2x2)

    x_4x4 = layers.Flatten()(x_4x4)
    x_4x4 = layers.Dropout(0.2)(x_4x4)

    x = layers.Concatenate()([x_1x1, x_2x2, x_4x4])

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((128, 1))(x)

    # Block 2
    x_1 = layers.Conv2D(32, (1, 1))(x)
    x_2 = layers.Conv2D(32, (1, 1))(x)
    x_2 = layers.Conv2D(32, (7, 1))(x_2)
    x_2 = layers.Conv2D(32, (1, 7))(x_2)
    x_3 = layers.Conv2D(32, (1, 1))(x)
    x_3 = layers.Conv2D(32, (7, 1))(x_3)
    x_3 = layers.Conv2D(32, (1, 7))(x_3)
    x_4 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x_4 = layers.Conv2D(32, (1, 1))(x_4)

    x_1 = layers.Activation('relu')(x_1)
    x_2 = layers.Activation('relu')(x_2)
    x_3 = layers.Activation('relu')(x_3)
    x_4 = layers.Activation('relu')(x_4)

    x = layers.Concatenate(axis=1)([x_1, x_2, x_3, x_4])

    # Output Layer
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=x)

    return model