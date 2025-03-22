import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))

    # Initial convolution
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # Parallel blocks
    block1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    block1 = layers.BatchNormalization()(block1)
    block2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    block2 = layers.BatchNormalization()(block2)
    block3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    block3 = layers.BatchNormalization()(block3)

    # Add outputs of blocks to initial convolution
    x = layers.Add()([x, block1, block2, block3])

    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
model = dl_model()
print(model.summary())