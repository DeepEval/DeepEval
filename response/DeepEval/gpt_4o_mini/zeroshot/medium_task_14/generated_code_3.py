import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1
    x1 = layers.Conv2D(32, (3, 3), padding='same')(input_layer)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    # Block 2
    x2 = layers.Conv2D(64, (3, 3), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    # Block 3
    x3 = layers.Conv2D(128, (3, 3), padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)

    # Output paths
    path1 = layers.GlobalAveragePooling2D()(x1)  # Pooling from Block 1
    path2 = layers.GlobalAveragePooling2D()(x2)  # Pooling from Block 2
    path3 = layers.GlobalAveragePooling2D()(x3)  # Pooling from Block 3

    # Parallel branch for direct processing
    x_direct = layers.Conv2D(64, (3, 3), padding='same')(input_layer)
    x_direct = layers.BatchNormalization()(x_direct)
    x_direct = layers.ReLU()(x_direct)
    x_direct = layers.GlobalAveragePooling2D()(x_direct)

    # Concatenate all paths
    concatenated = layers.concatenate([path1, path2, path3, x_direct])

    # Fully connected layers for classification
    x = layers.Dense(256, activation='relu')(concatenated)
    x = layers.Dropout(0.5)(x)  # Dropout for regularization
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()  # Print the model summary to verify the architecture