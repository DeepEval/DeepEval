import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

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

    # Parallel branch of convolutional layers
    parallel_branch = layers.Conv2D(64, (3, 3), padding='same')(input_layer)
    parallel_branch = layers.BatchNormalization()(parallel_branch)
    parallel_branch = layers.ReLU()(parallel_branch)

    # Concatenate outputs from all paths
    merged = layers.add([x1, x2, x3, parallel_branch])

    # Flatten the merged output
    flattened = layers.Flatten()(merged)

    # Fully connected layers
    dense1 = layers.Dense(256, activation='relu')(flattened)
    dense1 = layers.Dropout(0.5)(dense1)  # Adding dropout for regularization
    dense2 = layers.Dense(128, activation='relu')(dense1)
    output_layer = layers.Dense(10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Creating the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to instantiate the model
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # To display the model architecture