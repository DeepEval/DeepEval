import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # First block: Split input and apply depthwise separable convolutions
    splits = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Define different kernel sizes for depthwise separable convolutions
    conv_1x1 = layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(splits[0])
    conv_3x3 = layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu')(splits[1])
    conv_5x5 = layers.SeparableConv2D(32, kernel_size=(5, 5), activation='relu')(splits[2])

    # Batch normalization
    conv_1x1 = layers.BatchNormalization()(conv_1x1)
    conv_3x3 = layers.BatchNormalization()(conv_3x3)
    conv_5x5 = layers.BatchNormalization()(conv_5x5)

    # Concatenate outputs from the first block
    block_1_output = layers.Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5])

    # Second block: Multiple branches for feature extraction
    branch_1 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(block_1_output)
    branch_2 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(branch_1)

    branch_3 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(block_1_output)
    branch_3 = layers.Conv2D(32, kernel_size=(1, 7), activation='relu')(branch_3)
    branch_3 = layers.Conv2D(32, kernel_size=(7, 1), activation='relu')(branch_3)
    branch_3 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(branch_3)

    branch_4 = layers.AveragePooling2D(pool_size=(2, 2))(block_1_output)

    # Concatenate outputs from the second block
    block_2_output = layers.Concatenate(axis=-1)([branch_2, branch_3, branch_4])

    # Flatten the output for the fully connected layers
    flatten = layers.Flatten()(block_2_output)

    # Fully connected layers
    fc_1 = layers.Dense(128, activation='relu')(flatten)
    fc_1 = layers.BatchNormalization()(fc_1)
    fc_2 = layers.Dense(64, activation='relu')(fc_1)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(fc_2)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()