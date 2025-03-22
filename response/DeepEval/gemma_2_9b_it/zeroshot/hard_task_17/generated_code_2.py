import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Reshape((32, 32, 3))(x)
    block1_output = layers.multiply([inputs, x])

    # Block 2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(block1_output)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Fusion
    x = layers.Add()([block1_output, x])

    # Output Layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model