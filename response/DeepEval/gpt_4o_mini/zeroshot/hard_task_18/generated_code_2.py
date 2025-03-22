import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_tensor = layers.Input(shape=(32, 32, 3))

    # First block: Feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    # Skip connection: Adding input to the output of the first block
    x = layers.add([x, input_tensor])

    # Second block: Global average pooling and fully connected layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Reshape to generate channel weights
    channel_weights = layers.Reshape((1, 1, 32))(x)

    # Multiply weights by the input
    weighted_input = layers.multiply([input_tensor, channel_weights])

    # Flatten the output and add final classification layer
    flatten_output = layers.Flatten()(weighted_input)
    output_tensor = layers.Dense(10, activation='softmax')(flatten_output)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    return model

# Example usage:
model = dl_model()
model.summary()