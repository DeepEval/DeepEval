import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Initial convolutional layer to reduce dimensionality to 16
    x = layers.Conv2D(16, (3, 3), padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # First basic block
    main_path = layers.Conv2D(16, (3, 3), padding='same')(x)
    main_path = layers.BatchNormalization()(main_path)
    main_path = layers.ReLU()(main_path)

    # Branch for feature extraction
    branch = layers.Conv2D(16, (1, 1), padding='same')(x)

    # Feature fusion by adding both paths
    x = layers.Add()([main_path, branch])

    # Second basic block
    main_path = layers.Conv2D(16, (3, 3), padding='same')(x)
    main_path = layers.BatchNormalization()(main_path)
    main_path = layers.ReLU()(main_path)

    # Branch for feature extraction
    branch = layers.Conv2D(16, (1, 1), padding='same')(x)

    # Feature fusion by adding both paths again
    x = layers.Add()([main_path, branch])

    # Average pooling layer
    x = layers.AveragePooling2D(pool_size=(8, 8))(x)

    # Flatten the feature map
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    x = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=x)

    return model