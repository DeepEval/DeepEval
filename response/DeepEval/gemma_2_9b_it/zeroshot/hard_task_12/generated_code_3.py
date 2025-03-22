import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 64))

    # Main Path
    x = layers.Conv2D(filters=32, kernel_size=(1, 1))(inputs)
    
    # Parallel Convolutions
    conv1 = layers.Conv2D(filters=32, kernel_size=(1, 1))(x)
    conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3))(x)
    x = layers.concatenate([conv1, conv2], axis=-1)

    # Branch Path
    branch = layers.Conv2D(filters=32, kernel_size=(3, 3))(inputs)

    # Combine paths
    x = layers.add([x, branch])

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model