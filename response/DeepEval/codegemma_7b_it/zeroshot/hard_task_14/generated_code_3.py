from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    pooled_output = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(pooled_output)
    x = layers.Dense(32, activation='relu')(x)
    weight_layer = layers.Dense(32, activation='sigmoid', name='weights')

    # Branch path
    branch_input = layers.Input(shape=(32, 32, 3))
    branch_output = layers.Conv2D(32, (3, 3), activation='relu')(branch_input)

    # Combining outputs
    combined = layers.Add()([x, branch_output])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dense(10, activation='softmax')(x)

    # Create model
    model = models.Model(inputs=[inputs, branch_input], outputs=x)

    return model