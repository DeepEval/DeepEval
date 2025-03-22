import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 RGB images
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x1 = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    # Block 2
    x2 = layers.Conv2D(64, (3, 3), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.concatenate([x1, x2])  # Concatenate along the channel dimension

    # Block 3
    x3 = layers.Conv2D(128, (3, 3), padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.concatenate([x2, x3])  # Concatenate along the channel dimension

    # Flatten and Fully Connected Layers
    x = layers.Flatten()(x3)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=x)

    return model