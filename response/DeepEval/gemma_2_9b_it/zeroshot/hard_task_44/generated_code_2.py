import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(inputs)  
    x = [
        layers.Conv2D(32, kernel_size=(1, 1))(i) for i in x
    ] + [
        layers.Conv2D(32, kernel_size=(3, 3))(i) for i in x
    ] + [
        layers.Conv2D(32, kernel_size=(5, 5))(i) for i in x
    ]
    x = layers.Dropout(0.25)(tf.concat(x, axis=2))

    # Block 2
    x1 = layers.Conv2D(64, kernel_size=(1, 1))(x)
    x2 = layers.Conv2D(64, kernel_size=(1, 1))(x)
    x3 = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x3 = layers.Conv2D(64, kernel_size=(1, 1))(x3)
    x4 = layers.Conv2D(64, kernel_size=(5, 5))(x)
    
    x = tf.concat([x1, x2, x3, x4], axis=3)
    
    # Output
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model