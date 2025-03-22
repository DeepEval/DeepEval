from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer with dropout
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)

    # Second 1x1 convolutional layer with dropout
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # 3x1 convolutional layer with dropout
    x = layers.Conv2D(64, (3, 1), activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # 1x3 convolutional layer with dropout
    x = layers.Conv2D(64, (1, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Restore channels to match input
    x = layers.Conv2D(1, (1, 1), activation='relu')(x)

    # Add processed features to original input
    x = layers.Add()([x, inputs])

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model