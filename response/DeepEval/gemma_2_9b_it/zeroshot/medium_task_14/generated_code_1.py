import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Parallel branch
    x_parallel = layers.Conv2D(32, kernel_size=3, padding='same')(input_tensor)
    x_parallel = layers.BatchNormalization()(x_parallel)
    x_parallel = layers.ReLU()(x_parallel)

    # Block 1
    x_block1 = layers.Conv2D(32, kernel_size=3, padding='same')(input_tensor)
    x_block1 = layers.BatchNormalization()(x_block1)
    x_block1 = layers.ReLU()(x_block1)

    # Block 2
    x_block2 = layers.Conv2D(64, kernel_size=3, padding='same')(x_block1)
    x_block2 = layers.BatchNormalization()(x_block2)
    x_block2 = layers.ReLU()(x_block2)

    # Block 3
    x_block3 = layers.Conv2D(128, kernel_size=3, padding='same')(x_block2)
    x_block3 = layers.BatchNormalization()(x_block3)
    x_block3 = layers.ReLU()(x_block3)

    # Add outputs from all paths
    x = layers.add([x_parallel, x_block1, x_block2, x_block3])

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model