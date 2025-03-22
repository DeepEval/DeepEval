import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model


def dl_model():
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize image data
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Define initial convolution
    init_conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(train_images)

    # Define parallel blocks
    block1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(init_conv)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.ReLU()(block1)

    block2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(init_conv)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.ReLU()(block2)

    block3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(init_conv)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.ReLU()(block3)

    # Add outputs of parallel blocks to initial convolution
    concat = layers.Add()([init_conv, block1, block2, block3])

    # Flatten and pass through fully connected layers
    x = layers.Flatten()(concat)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=train_images, outputs=x)

    return model