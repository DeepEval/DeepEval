import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # 1x1 Convolutional Layer
    x = layers.Conv2D(32, kernel_size=1, activation='relu')(inputs)

    # Branch 1: Local Feature Extraction
    branch1 = layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    branch1 = layers.Conv2D(64, kernel_size=3, activation='relu')(branch1)

    # Branch 2: Downsampling and Upsampling
    branch2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(64, kernel_size=3, activation='relu')(branch2)
    branch2 = layers.Conv2DTranspose(64, kernel_size=2, strides=2, activation='relu')(branch2)

    # Branch 3: Downsampling and Upsampling
    branch3 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(64, kernel_size=3, activation='relu')(branch3)
    branch3 = layers.Conv2DTranspose(64, kernel_size=2, strides=2, activation='relu')(branch3)

    # Concatenate Branches
    x = layers.concatenate([branch1, branch2, branch3])

    # 1x1 Convolutional Layer for Refinement
    x = layers.Conv2D(128, kernel_size=1, activation='relu')(x)

    # Flatten and Fully Connected Layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model