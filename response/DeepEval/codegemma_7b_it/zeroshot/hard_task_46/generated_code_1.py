import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    inputs = keras.Input(shape=(32, 32, 3))

    # First Block
    x = tf.split(inputs, 3, axis=3)
    x = [layers.Lambda(lambda x: tf.keras.layers.SeparableConv2D(filters=32, kernel_size=k, padding="same", use_bias=False, activation="relu")(x))(i) for k, i in zip([1, 3, 5], x)]
    x = layers.concatenate(x)

    # Second Block
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    y = [
        layers.Conv2D(filters=32, kernel_size=1, padding="same", activation="relu")(x),
        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(y[0]),
        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(y[0]),
    ]
    y = layers.concatenate(y)
    z = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Concatenation
    x = layers.concatenate([x, y, z])

    # Global Average Pooling and Fully Connected Layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

model = dl_model()
model.summary()