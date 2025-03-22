import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_img = keras.Input(shape=(32, 32, 3))

    # Parallel branch
    x_parallel = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x_parallel = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x_parallel)

    # Block 1
    x_block1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x_block1 = layers.BatchNormalization()(x_block1)
    x_block1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x_block1)
    x_block1 = layers.BatchNormalization()(x_block1)

    # Block 2
    x_block2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x_block1)
    x_block2 = layers.BatchNormalization()(x_block2)
    x_block2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x_block2)
    x_block2 = layers.BatchNormalization()(x_block2)

    # Block 3
    x_block3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x_block2)
    x_block3 = layers.BatchNormalization()(x_block3)
    x_block3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x_block3)
    x_block3 = layers.BatchNormalization()(x_block3)

    # Concatenate outputs
    x = layers.concatenate([x_parallel, x_block1, x_block2, x_block3])

    # Classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  

    model = keras.Model(inputs=input_img, outputs=outputs)
    return model