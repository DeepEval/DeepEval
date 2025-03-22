import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    group1, group2, group3 = layers.Lambda(tf.split, axis=-1)([inputs, inputs, inputs])

    # Feature extraction using depthwise separable convolutional layers
    group1 = layers.DepthwiseConv2D(kernel_size=1, padding='same')(group1)
    group2 = layers.DepthwiseConv2D(kernel_size=3, padding='same')(group2)
    group3 = layers.DepthwiseConv2D(kernel_size=5, padding='same')(group3)

    # Concatenate the outputs of the three groups
    concat = layers.concatenate([group1, group2, group3])

    # Fusing features
    fusion = layers.Dense(512, activation='relu')(concat)

    # Classification layer
    outputs = layers.Dense(10, activation='softmax')(fusion)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Instantiate the model
model = dl_model()