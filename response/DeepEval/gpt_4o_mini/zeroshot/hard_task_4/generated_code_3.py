import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Step 1: Increase the dimensionality of input's channels threefold with 1x1 convolution
    x = layers.Conv2D(filters=3 * 3, kernel_size=(1, 1), padding='same', activation='relu')(inputs)

    # Step 2: Extract initial features using 3x3 depthwise separable convolution
    x = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)

    # Step 3: Compute channel attention weights through global average pooling
    channel_avg = layers.GlobalAveragePooling2D()(x)
    channel_dense_1 = layers.Dense(32, activation='relu')(channel_avg)
    channel_dense_2 = layers.Dense(3 * 3, activation='sigmoid')(channel_dense_1)

    # Step 4: Reshape weights to match initial features
    channel_weights = layers.Reshape((1, 1, 3 * 3))(channel_dense_2)

    # Step 5: Multiply with initial features to achieve channel attention weighting
    x = layers.multiply([x, channel_weights])

    # Step 6: 1x1 convolution to reduce dimensionality
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)

    # Step 7: Combine with initial input
    x = layers.add([x, inputs])

    # Step 8: Flatten and fully connected layer for classification
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()