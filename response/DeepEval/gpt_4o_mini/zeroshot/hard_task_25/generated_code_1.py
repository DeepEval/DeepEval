import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    inputs = layers.Input(shape=(32, 32, 3))

    # Main path
    # 1x1 Convolutional Layer
    main_path = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Split into three branches
    # Branch 1: Local feature extraction
    branch_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path)

    # Branch 2: Downsample, Process, and Upsample
    branch_2 = layers.AveragePooling2D(pool_size=(2, 2))(main_path)
    branch_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch_2)
    branch_2 = layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch_2)

    # Branch 3: Downsample, Process, and Upsample
    branch_3 = layers.AveragePooling2D(pool_size=(2, 2))(main_path)
    branch_3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch_3)
    branch_3 = layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch_3)

    # Concatenate all branches
    concatenated = layers.concatenate([branch_1, branch_2, branch_3])

    # 1x1 Convolutional Layer to form the main path output
    main_output = layers.Conv2D(filters=10, kernel_size=(1, 1), activation='relu')(concatenated)

    # Branch path
    branch_path = layers.Conv2D(filters=10, kernel_size=(1, 1), activation='relu')(inputs)

    # Fuse main path and branch path through addition
    merged = layers.add([main_output, branch_path])

    # Flatten and Fully Connected Layer for classification
    flattened = layers.Flatten()(merged)
    output = layers.Dense(units=10, activation='softmax')(flattened)

    # Create model
    model = models.Model(inputs=inputs, outputs=output)

    return model