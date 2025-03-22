import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Increase channel dimensionality threefold
    x = layers.Conv2D(filters=3 * 3, kernel_size=1, activation='relu')(input_tensor)

    # Extract initial features with depthwise separable convolution
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, activation='relu')(x)

    # Channel attention module
    channel_pool = layers.GlobalAveragePooling2D()(x)
    channel_dense1 = layers.Dense(units=128, activation='relu')(channel_pool)
    channel_dense2 = layers.Dense(units=3 * 3, activation='sigmoid')(channel_dense1)  
    channel_weights = layers.Reshape((3, 3, 1))(channel_dense2)
    x = layers.multiply([x, channel_weights])

    # Reduce dimensionality
    x = layers.Conv2D(filters=3, kernel_size=1, activation='relu')(x)

    # Add input to the output of channel attention
    x = layers.add([input_tensor, x])

    # Flatten and classify
    x = layers.Flatten()(x)
    output = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model