import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Conv2D(filters=64, kernel_size=(1, 1))(inputs)
    
    # Branch 1
    branch1 = layers.Conv2D(filters=64, kernel_size=(3, 3))(x)

    # Branch 2 & 3
    branch2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(filters=64, kernel_size=(3, 3))(branch2)
    branch2 = layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2))(branch2)

    branch3 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(filters=64, kernel_size=(3, 3))(branch3)
    branch3 = layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2))(branch3)

    # Concatenate branches
    x = layers.concatenate([branch1, branch2, branch3])
    x = layers.Conv2D(filters=64, kernel_size=(1, 1))(x)

    # Branch Path
    branch_path = layers.Conv2D(filters=64, kernel_size=(1, 1))(inputs)

    # Fuse outputs
    x = layers.add([x, branch_path])

    # Classification
    x = layers.Flatten()(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model