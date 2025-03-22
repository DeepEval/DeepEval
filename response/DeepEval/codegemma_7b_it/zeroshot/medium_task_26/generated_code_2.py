from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 64))

    # Compress input channels with 1x1 convolution
    x = layers.Conv2D(32, (1, 1), padding='same')(inputs)

    # Expand features through parallel convolutional layers
    branch1 = layers.Conv2D(64, (1, 1), padding='same')(x)
    branch2 = layers.Conv2D(64, (1, 1), padding='same')(x)
    branch2 = layers.Conv2D(64, (3, 3), padding='same')(branch2)

    # Concatenate results
    merged = layers.Concatenate()([branch1, branch2])

    # Flatten and pass through fully connected layers
    flattened = layers.Flatten()(merged)
    outputs = layers.Dense(10, activation='softmax')(flattened)

    # Create model
    model = models.Model(inputs, outputs)

    return model