import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(28, 28, 1))  

    # Block 1
    x = layers.Conv2D(32, kernel_size=1, activation='relu')(inputs)
    x = layers.DepthwiseConv2D(kernel_size=3, activation='relu')(x)
    x = layers.Conv2D(32, kernel_size=1, activation='relu')(x)

    branch = layers.DepthwiseConv2D(kernel_size=3, activation='relu')(inputs)
    branch = layers.Conv2D(32, kernel_size=1, activation='relu')(branch)

    x = layers.Concatenate(axis=-1)([x, branch])

    # Block 2
    shape = layers.Lambda(lambda x: tf.shape(x), name='shape')(x)
    x = layers.Reshape((shape[1], shape[2], 4, 32))(x)
    x = layers.Permute((2, 3, 1, 4))(x)
    x = layers.Reshape((shape[1], shape[2], 128))(x)

    # Output Layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model