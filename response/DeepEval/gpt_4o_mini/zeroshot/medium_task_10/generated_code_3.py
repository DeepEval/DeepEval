import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def basic_block(x, filters):
    # Main path
    main_path = layers.Conv2D(filters, (3, 3), padding='same')(x)
    main_path = layers.BatchNormalization()(main_path)
    main_path = layers.ReLU()(main_path)

    # Branch
    branch = x
    
    # Combine paths
    x = layers.add([main_path, branch])
    x = layers.ReLU()(x)
    
    return x

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image dimensions
    inputs = layers.Input(shape=input_shape)

    # First level
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = basic_block(x, 16)  # First basic block

    # Second level with two residual blocks
    for _ in range(2):
        x = basic_block(x, 16)

    # Additional branch for the second level
    branch = layers.Conv2D(16, (1, 1), padding='same')(inputs)  # Independent convolution
    x = layers.add([x, branch])  # Combine paths of the second level
    x = layers.ReLU()(x)

    # Third level
    global_branch = layers.Conv2D(16, (1, 1), padding='same')(inputs)  # Global branch
    x = layers.GlobalAveragePooling2D()(x)  # Pooling after second level
    x = layers.add([x, global_branch])  # Combine global branch with second level output
    
    # Fully connected layer for classification
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    model = models.Model(inputs=inputs, outputs=x)
    
    return model

# Example usage
model = dl_model()
model.summary()