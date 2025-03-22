import keras
from keras import layers

def dl_model():

    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    x3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    x1 = layers.Flatten()(x1)
    x2 = layers.Flatten()(x2)
    x3 = layers.Flatten()(x3)

    x1 = layers.Dropout(0.4)(x1)
    x2 = layers.Dropout(0.4)(x2)
    x3 = layers.Dropout(0.4)(x3)

    x = layers.concatenate([x1, x2, x3])

    # Block 2
    x = layers.Reshape((-1, 1, 1))(x)
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Reshape((-1, 4, 4))(x)

    b1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    b2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    b2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(b2)
    b3 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    b3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(b3)
    b3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(b3)
    b4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    b4 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(b4)

    x = layers.concatenate([b1, b2, b3, b4])

    # Output layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=x)

    return model