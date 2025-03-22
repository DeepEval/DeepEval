import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same', depth_wise=True)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(32, (1, 1), padding='same')(x)
    x = layers.Conv2D(32, (1, 1), padding='same')(x)

    # Branch Path
    branch_x = layers.Conv2D(32, (1, 1), padding='same')(inputs)

    # Combine Outputs
    x = layers.Add()([x, branch_x])

    # Flatten and Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model