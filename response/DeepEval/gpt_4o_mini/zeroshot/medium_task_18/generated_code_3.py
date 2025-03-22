import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # 1x1 Convolution
    conv1x1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # 3x3 Convolution
    conv3x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # 5x5 Convolution
    conv5x5 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)

    # 3x3 Max Pooling
    pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)

    # Concatenate the features
    concatenated = layers.Concatenate()([conv1x1, conv3x3, conv5x5, pool])

    # Add a few more convolutional layers
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concatenated)
    x = layers.MaxPooling2D((2, 2))(x)  # 2x2 Max Pooling
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 2x2 Max Pooling

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=x)

    return model

# Example usage
model = dl_model()
model.summary()