import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3)) 

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    x = [layers.Conv2D(filters=inputs.shape[-1]//3, kernel_size=(1,1))(xi) for xi in x]
    x = layers.Concatenate(axis=3)(x)

    # Block 2
    shape = layers.Lambda(lambda x: tf.shape(x))(x)
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, x[0], x[1], 3, x[2]//3)))([shape] + [x])
    x = layers.Lambda(lambda x: tf.transpose(x, [0, 1, 2, 4, 3]))(x)
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, x[0], x[1], x[2] * x[3])))(x)

    # Block 3
    x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", depth_wise=True, activation="relu")(x)

    # Branch
    branch = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(inputs)

    # Combine outputs
    x = layers.add([x, branch])

    # Final layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(units=10, activation="softmax")(x) 

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model