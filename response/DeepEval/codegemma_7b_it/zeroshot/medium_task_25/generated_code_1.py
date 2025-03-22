import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path1 = layers.Conv2D(64, (1, 1), padding='same')(input_layer)

    # Path 2: Average pooling followed by 1x1 convolution
    path2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path2 = layers.Conv2D(64, (1, 1), padding='same')(path2)

    # Path 3: 1x1 convolution followed by parallel 1x3 and 3x1 convolutions
    path3 = layers.Conv2D(64, (1, 1), padding='same')(input_layer)
    path3_1x3 = layers.Conv2D(64, (1, 3), padding='same')(path3)
    path3_3x1 = layers.Conv2D(64, (3, 1), padding='same')(path3)
    path3 = layers.Concatenate(axis=-1)([path3_1x3, path3_3x1])

    # Path 4: 1x1 convolution followed by 3x3 convolution, then parallel 1x3 and 3x1 convolutions
    path4 = layers.Conv2D(64, (1, 1), padding='same')(input_layer)
    path4 = layers.Conv2D(64, (3, 3), padding='same')(path4)
    path4_1x3 = layers.Conv2D(64, (1, 3), padding='same')(path4)
    path4_3x1 = layers.Conv2D(64, (3, 1), padding='same')(path4)
    path4 = layers.Concatenate(axis=-1)([path4_1x3, path4_3x1])

    # Multi-scale feature fusion
    fusion = layers.Concatenate(axis=-1)([path1, path2, path3, path4])

    # Fully connected layer for classification
    output_layer = layers.Dense(10, activation='softmax')(fusion)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.summary()