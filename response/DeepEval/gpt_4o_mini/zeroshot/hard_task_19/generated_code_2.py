import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for the model
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Branch path
    y = layers.GlobalAveragePooling2D()(input_tensor)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(64, activation='relu')(y)
    channel_weights = layers.Dense(32 * 32 * 3, activation='sigmoid')(y)  # Output layer for channel weights
    channel_weights = layers.Reshape((32, 32, 3))(channel_weights)

    # Multiply channel weights with input
    weighted_input = layers.multiply([input_tensor, channel_weights])

    # Adding both paths
    combined = layers.add([x, weighted_input])

    # Classification path
    z = layers.Flatten()(combined)
    z = layers.Dense(256, activation='relu')(z)
    z = layers.Dense(128, activation='relu')(z)
    output_tensor = layers.Dense(10, activation='softmax')(z)  # CIFAR-10 has 10 classes

    # Constructing the model
    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    return model

# Example of creating the model
model = dl_model()
model.summary()