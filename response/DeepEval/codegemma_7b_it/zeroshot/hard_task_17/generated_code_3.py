import tensorflow as tf
from tensorflow.keras import layers

def dl_model():

    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1: Global Average Pooling and Fully Connected Layers
    block1 = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    block1 = layers.Conv2D(64, (3, 3), activation='relu')(block1)
    block1 = layers.GlobalAveragePooling2D()(block1)
    block1 = layers.Dense(64, activation='relu')(block1)
    block1 = layers.Dense(64, activation='relu')(block1)

    # Reshape block1 weights to match input shape
    block1_weights = layers.Reshape((32, 32, 64))(block1)

    # Block 2: Convolutional Layers and Max Pooling
    block2 = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    block2 = layers.Conv2D(64, (3, 3), activation='relu')(block2)
    block2 = layers.MaxPooling2D()(block2)

    # Branch from Block 1 to Block 2 output
    branch = layers.Multiply()([block1_weights, block2])

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(branch)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model