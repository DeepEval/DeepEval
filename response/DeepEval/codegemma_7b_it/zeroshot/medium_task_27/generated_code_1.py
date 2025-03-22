import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Create two parallel convolutional branches
    branch_a = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    branch_b = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)

    # Combine the branches through addition
    merged = layers.add([branch_a, branch_b])

    # Global average pooling
    pooled = layers.GlobalAveragePooling2D()(merged)

    # Fully connected layers for attention weights
    attention_a = layers.Dense(32, activation='relu')(pooled)
    attention_b = layers.Dense(32, activation='relu')(pooled)

    # Output layer for attention weights
    attention_weights = layers.Dense(10, activation='softmax')(attention_b)

    # Weighted output
    weighted_output = layers.multiply([merged, attention_weights])
    weighted_output = layers.GlobalAveragePooling2D()(weighted_output)

    # Final output layer
    outputs = layers.Dense(10, activation='softmax')(weighted_output)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model