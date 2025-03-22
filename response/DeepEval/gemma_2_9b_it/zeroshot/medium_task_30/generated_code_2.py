import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))  

    # Layer 1: Average pooling with 1x1 window and stride 1
    x1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)

    # Layer 2: Average pooling with 2x2 window and stride 2
    x2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)

    # Layer 3: Average pooling with 4x4 window and stride 4
    x3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Concatenate the outputs of the pooling layers
    x = layers.Concatenate()([x1, x2, x3])

    # Flatten the concatenated output
    x = layers.Flatten()(x)

    # Fully connected layer 1
    x = layers.Dense(128, activation='relu')(x)

    # Fully connected layer 2
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model