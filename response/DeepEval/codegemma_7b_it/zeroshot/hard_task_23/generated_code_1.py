import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer for CIFAR-10 images
    inputs = keras.Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    x = layers.Conv2D(filters=64, kernel_size=1, padding='same')(inputs)

    # Local feature extraction branch
    branch_1 = layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
    branch_1 = layers.Conv2D(filters=64, kernel_size=3, padding='same')(branch_1)

    # Downsampling branch 1
    branch_2 = layers.AveragePooling2D(pool_size=2)(x)
    branch_2 = layers.Conv2D(filters=64, kernel_size=3, padding='same')(branch_2)

    # Upsampling branch 1
    branch_2 = layers.UpSampling2D(size=2)(branch_2)

    # Downsampling branch 2
    branch_3 = layers.AveragePooling2D(pool_size=2)(x)
    branch_3 = layers.Conv2D(filters=64, kernel_size=3, padding='same')(branch_3)
    branch_3 = layers.UpSampling2D(size=2)(branch_3)

    # Concatenate branches and refine output
    merged = layers.concatenate([branch_1, branch_2, branch_3])
    outputs = layers.Conv2D(filters=10, kernel_size=1, padding='same')(merged)

    # Flatten and softmax for classification
    outputs = layers.Flatten()(outputs)
    outputs = layers.Activation('softmax')(outputs)

    # Create and return the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model