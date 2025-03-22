import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 channels
    inputs = layers.Input(shape=input_shape)

    # Step 1: 1x1 Convolution to increase channel dimensionality
    x = layers.Conv2D(9, kernel_size=(1, 1), padding='same')(inputs)  # Increase channels to 9

    # Step 2: 3x3 Depthwise Separable Convolution
    x = layers.SeparableConv2D(filters=9, kernel_size=(3, 3), padding='same')(x)

    # Step 3: Global Average Pooling to compute channel attention weights
    gap = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers to generate channel attention weights
    x = layers.Dense(18, activation='relu')(gap)  # FC layer to increase to 18
    x = layers.Dense(9, activation='sigmoid')(x)  # FC layer to match number of channels

    # Reshape weights to match the initial feature dimensions
    channel_weights = layers.Reshape((1, 1, 9))(x)

    # Step 4: Multiply initial features with channel weights for attention
    x = layers.multiply([x, channel_weights])

    # Step 5: 1x1 Convolution to reduce dimensionality
    x = layers.Conv2D(3, kernel_size=(1, 1), padding='same')(x)  # Reduce back to 3 channels

    # Combine output with initial input
    x = layers.add([x, inputs])  # Residual connection

    # Step 6: Flatten and apply final classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)  # Fully connected layer
    outputs = layers.Dense(10, activation='softmax')(x)  # Output layer for 10 classes

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Display the model architecture