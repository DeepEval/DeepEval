import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image dimensions
    inputs = layers.Input(shape=input_shape)

    # First block: feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    # Adding the input to the output of the first block
    x_shortcut = layers.AveragePooling2D(pool_size=(2, 2))(inputs)  # Adjust input size
    x = layers.add([x, x_shortcut])

    # Second block: compressing the feature map
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Reshape for multiplication
    channel_weights = layers.Reshape((1, 1, 32))(x)

    # Multiply the input by the channel weights
    x = layers.multiply([inputs, channel_weights])

    # Flatten and classification layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model