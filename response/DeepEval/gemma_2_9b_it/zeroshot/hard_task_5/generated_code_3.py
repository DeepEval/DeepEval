import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
    x = [layers.Conv2D(filters=32 // 3, kernel_size=1, padding='same')(xi) for xi in x]
    x = layers.Concatenate(axis=-1)(x)

    # Block 2
    shape = layers.Lambda(lambda x: tf.shape(x))(x)
    x = layers.Lambda(lambda x: tf.reshape(x, (*x[:3], 3, x[3] // 3)))(x)
    x = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2]))(x)
    x = layers.Lambda(lambda x: tf.reshape(x, (*x[:3], x[3] * 3)))(x)

    # Block 3
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    branch = layers.Lambda(lambda x: x)(input_tensor)
    x = layers.Add()([x, branch])

    # Classification head
    x = layers.Flatten()(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    return model