from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # First block
    # Main path
    main_path = layers.Conv2D(filters=16, kernel_size=3, activation='relu')(inputs)
    main_path = layers.Conv2D(filters=16, kernel_size=3, activation='relu')(main_path)
    main_path = layers.Conv2D(filters=3, kernel_size=1)(main_path)

    # Branch path
    branch_path = layers.Conv2D(filters=16, kernel_size=1)(inputs)

    # Combine paths
    combined = layers.Add()([main_path, branch_path])

    # Second block
    # Split into three groups
    x1, x2, x3 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(combined)

    # Extract features with depthwise separable convolutions
    x1 = layers.SeparableConv2D(filters=3, kernel_size=1, activation='relu')(x1)
    x2 = layers.SeparableConv2D(filters=3, kernel_size=3, activation='relu')(x2)
    x3 = layers.SeparableConv2D(filters=3, kernel_size=5, activation='relu')(x3)

    # Concatenate groups
    concat = layers.Concatenate()([x1, x2, x3])

    # Fully connected layers
    flatten = layers.Flatten()(concat)
    outputs = layers.Dense(units=10, activation='softmax')(flatten)

    # Model creation
    model = models.Model(inputs=inputs, outputs=outputs)

    return model