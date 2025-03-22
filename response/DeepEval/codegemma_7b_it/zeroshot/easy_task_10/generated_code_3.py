from tensorflow.keras import layers, models

def dl_model():
    inputs = layers.Input(shape=(28, 28, 1))

    # Increase dimensionality with a 1x1 convolutional layer
    x = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(inputs)

    # Feature extraction with 3x3 depthwise separable convolutional layer
    x = layers.SeparableConv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)

    # Reduce dimensionality with a 1x1 convolutional layer with stride 2
    x = layers.Conv2D(filters=16, kernel_size=1, strides=2, activation='relu')(x)

    # Flatten and fully connected layer for classification
    x = layers.Flatten()(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model