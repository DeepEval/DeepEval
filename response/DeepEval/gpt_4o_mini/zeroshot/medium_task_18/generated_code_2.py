import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # 1x1 Convolution
    conv_1x1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # 3x3 Convolution
    conv_3x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # 5x5 Convolution
    conv_5x5 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)

    # 3x3 Max Pooling
    max_pooling = layers.MaxPooling2D(pool_size=(3, 3), padding='same')(inputs)

    # Concatenate the feature maps
    concatenated = layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5, max_pooling])

    # Flatten the concatenated feature maps
    flattened = layers.Flatten()(concatenated)

    # Fully connected layers
    dense_1 = layers.Dense(128, activation='relu')(flattened)
    dense_2 = layers.Dense(10, activation='softmax')(dense_1)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=dense_2)

    return model

# Example usage
model = dl_model()
model.summary()