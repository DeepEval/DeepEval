from tensorflow.keras import layers, Model

def dl_model():
    inputs = layers.Input(shape=(224, 224, 3))

    # First sequential feature extraction layer
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = layers.AveragePooling2D()(x)

    # Second sequential feature extraction layer
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.AveragePooling2D()(x)

    # Third convolutional layer
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)

    # Fourth convolutional layer
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)

    # Fifth convolutional layer
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.AveragePooling2D()(x)

    # Flatten the feature maps
    x = layers.Flatten()(x)

    # First fully connected layer with dropout
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)

    # Second fully connected layer with dropout
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)

    # Output layer with softmax activation for 1000 classes
    outputs = layers.Dense(units=1000, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

model = dl_model()