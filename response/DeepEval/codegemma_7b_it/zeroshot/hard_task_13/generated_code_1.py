import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1: Parallel branches for feature extraction
    branch_1 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
    branch_2 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    branch_3 = layers.Conv2D(64, (5, 5), padding='same')(inputs)
    branch_4 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inputs)

    # Concatenate outputs from parallel branches
    concat = layers.concatenate([branch_1, branch_2, branch_3, branch_4])

    # Block 2: Global average pooling and fully connected layers
    gap = layers.GlobalAveragePooling2D()(concat)
    fc_1 = layers.Dense(512, activation='relu')(gap)
    fc_2 = layers.Dense(256, activation='relu')(fc_1)

    # Generate weights for feature-wise multiplication
    weights = layers.Dense(64)(fc_2)
    weights = layers.Reshape((1, 1, 64))(weights)

    # Element-wise multiplication and final fully connected layer
    output = layers.multiply([concat, weights])
    output = layers.Conv2D(10, (1, 1), padding='same')(output)
    output = layers.GlobalAveragePooling2D()(output)
    output = layers.Dense(10, activation='softmax')(output)

    # Create the model
    model = models.Model(inputs=inputs, outputs=output)

    return model