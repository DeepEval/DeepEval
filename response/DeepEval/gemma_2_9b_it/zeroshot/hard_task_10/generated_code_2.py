from tensorflow.keras import layers, models

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    x1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Path 2: Sequence of convolutions
    x2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    x2 = layers.Conv2D(32, (1, 7), activation='relu')(x2)
    x2 = layers.Conv2D(32, (7, 1), activation='relu')(x2)

    # Concatenate outputs from both paths
    x = layers.concatenate([x1, x2])
    x = layers.Conv2D(3, (1, 1), activation='relu')(x)

    # Branch connecting directly to input
    branch = layers.Conv2D(3, (1, 1), activation='relu')(inputs)

    # Merge outputs through addition
    x = layers.add([x, branch])

    # Classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model