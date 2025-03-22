import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = layers.Input(shape=(32, 32, 3))

    # Compress input features with global average pooling
    avg_pool = layers.GlobalAveragePooling2D()(input_layer)

    # Fully connected layer to generate weights
    fc1 = layers.Dense(32, activation='relu')(avg_pool)

    # Fully connected layer to obtain final probability distribution
    output_layer = layers.Dense(10, activation='softmax')(fc1)

    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model