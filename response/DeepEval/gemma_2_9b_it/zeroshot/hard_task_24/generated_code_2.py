import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Branch 1: Local Features
    x1 = layers.Conv2D(32, kernel_size=1, activation='relu')(inputs)
    x1 = layers.Conv2D(32, kernel_size=3, activation='relu')(x1)

    # Branch 2: Downsampling & Upsampling
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(inputs)
    x2 = layers.Conv2D(64, kernel_size=3, activation='relu')(x2)
    x2 = layers.UpSampling2D(size=(2, 2))(x2)

    # Branch 3: Downsampling & Upsampling
    x3 = layers.MaxPooling2D(pool_size=(2, 2))(inputs)
    x3 = layers.Conv2D(64, kernel_size=3, activation='relu')(x3)
    x3 = layers.UpSampling2D(size=(2, 2))(x3)

    # Concatenate branches
    merged = layers.Concatenate()([x1, x2, x3])

    # Final 1x1 Convolution
    merged = layers.Conv2D(128, kernel_size=1, activation='relu')(merged)

    # Flatten and Fully Connected Layers
    x = layers.Flatten()(merged)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model