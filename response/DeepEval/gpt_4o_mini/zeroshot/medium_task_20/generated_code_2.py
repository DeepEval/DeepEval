import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Path 1: 1x1 Convolution
    path1 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Path 2: 1x1 Convolution followed by two stacked 3x3 Convolutions
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    path2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(path2)
    path2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(path2)

    # Path 3: 1x1 Convolution followed by a single 3x3 Convolution
    path3 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    path3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(path3)

    # Path 4: Max Pooling followed by 1x1 Convolution
    path4 = layers.MaxPooling2D(pool_size=(2, 2))(input_layer)
    path4 = layers.Conv2D(32, (1, 1), activation='relu')(path4)

    # Concatenate all paths
    concatenated = layers.concatenate([path1, path2, path3, path4])

    # Flatten the concatenated outputs
    flatten = layers.Flatten()(concatenated)

    # Dense layer with 128 units
    dense = layers.Dense(128, activation='relu')(flatten)

    # Output layer with softmax activation for 10 categories
    output_layer = layers.Dense(10, activation='softmax')(dense)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.summary()  # To view the model architecture