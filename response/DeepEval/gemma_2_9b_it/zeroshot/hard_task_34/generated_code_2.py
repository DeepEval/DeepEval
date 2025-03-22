import tensorflow as tf
from tensorflow import keras

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Main path
    x = inputs

    for _ in range(3):
        x = keras.layers.ReLU()(keras.layers.SeparableConv2D(32, (3, 3), padding='same')(x))
        
        # Concatenate features
        x = keras.layers.concatenate([x, keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)], axis=3)

    # Branch path
    branch_path = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)

    # Fuse features
    x = keras.layers.add([x, branch_path])

    # Flatten and output layer
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model