import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image shape
    inputs = layers.Input(shape=input_shape)

    # Path 1: 1x1 Convolution
    path1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Path 2: Sequence of convolutions (1x1, 1x7, 7x1)
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path2 = layers.Conv2D(32, (1, 7), activation='relu')(path2)
    path2 = layers.Conv2D(32, (7, 1), activation='relu')(path2)

    # Concatenate the outputs of both paths
    concatenated = layers.concatenate([path1, path2])

    # 1x1 Convolution to align the output dimensions
    main_output = layers.Conv2D(32, (1, 1), activation='relu')(concatenated)

    # Branch connection directly to the input
    branch = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Merge outputs of the main path and branch via addition
    merged = layers.add([main_output, branch])

    # Global average pooling to reduce spatial dimensions
    pooled_output = layers.GlobalAveragePooling2D()(merged)

    # Fully connected layers for classification
    dense1 = layers.Dense(128, activation='relu')(pooled_output)
    dense2 = layers.Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=dense2)

    return model

# Example of how to create the model
model = dl_model()
model.summary()