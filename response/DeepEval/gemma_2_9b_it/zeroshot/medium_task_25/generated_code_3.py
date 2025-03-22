from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))  

    # Path 1
    x1 = layers.Conv2D(32, 1, activation='relu')(inputs)

    # Path 2
    x2 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
    x2 = layers.Conv2D(32, 1, activation='relu')(x2)

    # Path 3
    x3 = layers.Conv2D(32, 1, activation='relu')(inputs)
    x3_1 = layers.Conv2D(32, 1, activation='relu')(x3)
    x3_2 = layers.Conv2D(32, 3, activation='relu', padding='same')(x3)
    x3 = layers.concatenate([x3_1, x3_2], axis=-1)

    # Path 4
    x4 = layers.Conv2D(32, 1, activation='relu')(inputs)
    x4 = layers.Conv2D(32, 3, activation='relu', padding='same')(x4)
    x4_1 = layers.Conv2D(32, 1, activation='relu')(x4)
    x4_2 = layers.Conv2D(32, 3, activation='relu', padding='same')(x4)
    x4 = layers.concatenate([x4_1, x4_2], axis=-1)

    # Feature Fusion
    outputs = layers.concatenate([x1, x2, x3, x4], axis=-1)

    # Classification
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model