import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for images with shape (32, 32, 3)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Branch path
    y = layers.GlobalAveragePooling2D()(input_layer)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(32, activation='relu')(y)
    
    # Generate channel weights
    channel_weights = layers.Reshape((1, 1, 32))(y)

    # Multiply the input with the channel weights
    weighted_input = layers.multiply([input_layer, channel_weights])

    # Add outputs from main path and weighted input
    merged = layers.add([x, weighted_input])

    # Fully connected layers for classification
    flattened = layers.Flatten()(merged)
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    output_layer = layers.Dense(10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# To use the model, you can call:
model = dl_model()
model.summary()  # To see the model architecture