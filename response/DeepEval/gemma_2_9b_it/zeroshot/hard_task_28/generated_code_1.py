import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x) 
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x) 

    # Branch Path
    branch_x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(inputs)

    # Combine paths
    x = layers.add([x, branch_x])

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model