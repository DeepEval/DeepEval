import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Parallel convolutional branches
    branch_3x3 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    branch_5x5 = layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(input_layer)

    # Global average pooling
    avg_pool = layers.GlobalAveragePooling2D()(branch_3x3 + branch_5x5)

    # Fully connected layers for attention weights
    attention_fc = layers.Dense(units=10, activation='softmax')(avg_pool)

    # Weighted output
    weighted_branch_3x3 = layers.multiply([branch_3x3, attention_fc])
    weighted_branch_5x5 = layers.multiply([branch_5x5, attention_fc])
    weighted_output = layers.add([weighted_branch_3x3, weighted_branch_5x5])

    # Output layer
    output_layer = layers.Dense(units=10, activation='softmax')(weighted_output)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model