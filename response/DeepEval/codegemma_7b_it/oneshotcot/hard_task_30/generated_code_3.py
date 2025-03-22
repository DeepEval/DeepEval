import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # First block: Dual-path architecture
    # Main path
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(16, (1, 1), padding='same')(x)

    # Branch path
    shortcut = layers.Conv2D(16, (1, 1), padding='same')(inputs)

    # Combine paths
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    # Second block: Grouped depthwise separable convolutions
    x = layers.Reshape((-1, 32))(x)  # Flatten the spatial dimensions for tf.split
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=1))(x)  # Split into three groups

    # Grouped convolutions
    conv1 = layers.Lambda(lambda x: layers.Conv2D(16, (1, 1), padding='same')(x))(x[0])
    conv2 = layers.Lambda(lambda x: layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x))(x[1])
    conv3 = layers.Lambda(lambda x: layers.DepthwiseConv2D(kernel_size=(5, 5), padding='same')(x))(x[2])

    # Concatenate outputs
    x = layers.concatenate([conv1, conv2, conv3])

    # Fully connected layers
    x = layers.Reshape((-1, 48))(x)  # Reshape for the dense layers
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model