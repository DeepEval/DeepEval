# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

# Define the dl_model function
def dl_model():
    # Input layer with shape 32x32x3
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1: Feature Extraction
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x = layers.DepthwiseConv2D(kernel_size=1, padding='same', name='DWConv1')(x[0])
    x = layers.Conv2D(6, kernel_size=1, padding='same', name='Conv1')(x)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', name='DWConv2')(x[1])
    x = layers.Conv2D(6, kernel_size=1, padding='same', name='Conv2')(x)
    x = layers.DepthwiseConv2D(kernel_size=5, padding='same', name='DWConv3')(x[2])
    x = layers.Conv2D(6, kernel_size=1, padding='same', name='Conv3')(x)
    x = layers.Concatenate(axis=-1)([x[0], x[1], x[2]])

    # Block 2: Feature Extraction
    y = layers.Conv2D(6, kernel_size=1, padding='same', activation='relu')(x)
    y = layers.DepthwiseConv2D(kernel_size=3, padding='same', activation='relu')(y)
    y = layers.Conv2D(12, kernel_size=1, padding='same')(y)
    z = layers.Conv2D(6, kernel_size=1, padding='same', activation='relu')(x)
    z = layers.DepthwiseConv2D(kernel_size=3, padding='same', activation='relu')(z)
    z = layers.Conv2D(12, kernel_size=1, padding='same')(z)
    w = layers.MaxPooling2D(pool_size=2, padding='same')(x)
    w = layers.Conv2D(6, kernel_size=1, padding='same')(w)
    u = layers.Concatenate(axis=-1)([y, z, w])

    # Concatenate feature maps from Block 1 and Block 2
    x = layers.Concatenate(axis=-1)([x, u])

    # Flatten and Dense layers
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Usage
from tensorflow import keras
from tensorflow.keras import backend as tf
model = dl_model()
model.summary()