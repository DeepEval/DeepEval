import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
    inputs = layers.Input(shape=input_shape)

    # Branch 1: 3x3 Convolutions
    branch1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    branch1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch1)

    # Branch 2: 1x1 Convolution followed by two 3x3 Convolutions
    branch2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: Max Pooling
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(inputs)

    # Concatenating the outputs of all branches
    merged = layers.concatenate([branch1, branch2, branch3])

    # Flatten the merged output
    flattened = layers.Flatten()(merged)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=dense2)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Display the model architecture