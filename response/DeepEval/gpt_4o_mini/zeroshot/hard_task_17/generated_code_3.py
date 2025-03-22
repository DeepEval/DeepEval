import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Reshape and multiply with input
    x_reshaped = layers.Reshape((1, 1, 32))(x)  # Reshape to match input shape for multiplication
    weighted_features = layers.Multiply()([inputs, x_reshaped])

    # Block 2
    y = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(weighted_features)
    y = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(y)
    y = layers.MaxPooling2D(pool_size=(2, 2))(y)

    # Branch from Block 1
    block_1_output = layers.GlobalAveragePooling2D()(weighted_features)
    block_1_output = layers.Dense(32, activation='relu')(block_1_output)

    # Fuse outputs from Block 1 and Block 2
    y_flattened = layers.GlobalAveragePooling2D()(y)
    combined = layers.Add()([block_1_output, y_flattened])

    # Final classification layers
    combined = layers.Dense(64, activation='relu')(combined)
    outputs = layers.Dense(num_classes, activation='softmax')(combined)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
model.summary()