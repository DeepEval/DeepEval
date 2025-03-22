from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Branch 1: Local Features
    x1 = layers.Conv2D(32, 1, activation='relu')(inputs)
    x1 = layers.Conv2D(32, 3, activation='relu')(x1)

    # Branch 2: Downsampling & Upsampling
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(inputs)
    x2 = layers.Conv2D(64, 3, activation='relu')(x2)
    x2 = layers.UpSampling2D((2, 2))(x2)

    # Branch 3: Downsampling & Upsampling
    x3 = layers.MaxPooling2D(pool_size=(2, 2))(inputs)
    x3 = layers.Conv2D(64, 3, activation='relu')(x3)
    x3 = layers.UpSampling2D((2, 2))(x3)

    # Concatenate branches
    x = layers.concatenate([x1, x2, x3])
    x = layers.Conv2D(128, 1, activation='relu')(x)

    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model