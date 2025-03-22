from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Path 1
    x1 = layers.Conv2D(32, 1, activation='relu')(input_tensor)

    # Path 2
    x2 = layers.Conv2D(32, 1, activation='relu')(input_tensor)
    x2 = layers.Conv2D(32, 7, activation='relu', padding='same')(x2)
    x2 = layers.Conv2D(32, 1, activation='relu', padding='same')(x2)

    # Path 3
    x3 = layers.Conv2D(32, 1, activation='relu')(input_tensor)
    x3_1 = layers.Conv2D(32, 7, activation='relu', padding='same')(x3)
    x3_2 = layers.Conv2D(32, 1, activation='relu', padding='same')(x3)
    x3_3 = layers.Conv2D(32, 7, activation='relu', padding='same')(x3_2)
    x3 = layers.concatenate([x3_1, x3_3], axis=3)

    # Path 4
    x4 = layers.AveragePooling2D(pool_size=(2, 2))(input_tensor)
    x4 = layers.Conv2D(32, 1, activation='relu')(x4)

    # Concatenate outputs
    x = layers.concatenate([x1, x2, x3, x4], axis=3)

    # Flatten and classify
    x = layers.Flatten()(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output)

    return model