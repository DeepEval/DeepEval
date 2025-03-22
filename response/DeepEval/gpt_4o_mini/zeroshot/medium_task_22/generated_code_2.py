import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Branch 1: 3x3 Convolutions
    branch1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    branch1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch1)

    # Branch 2: 1x1 Conv followed by two 3x3 Convolutions
    branch2 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: Max Pooling
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(inputs)

    # Concatenate the outputs of the three branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # Flatten the concatenated feature maps
    flattened = layers.Flatten()(concatenated)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(64, activation='relu')(dense1)

    # Output layer with softmax activation for classification (10 classes for CIFAR-10)
    outputs = layers.Dense(10, activation='softmax')(dense2)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()