import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Initial Convolution
    x = layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(inputs)

    # Basic Block
    def basic_block(x):
        identity = x
        x = layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=16, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, identity])
        return x

    # Level 1
    x = basic_block(x)

    # Level 2
    def residual_block(x):
        identity = x
        x = basic_block(x)
        branch = layers.Conv2D(filters=16, kernel_size=1, padding='same')(identity)
        branch = layers.BatchNormalization()(branch)
        x = layers.add([x, branch])
        return x

    x = residual_block(x)
    x = residual_block(x)

    # Level 3
    global_branch = layers.Conv2D(filters=16, kernel_size=1, padding='same')(x)
    global_branch = layers.BatchNormalization()(global_branch)
    x = layers.add([x, global_branch])

    # Average Pooling and Classification
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model