import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for 32x32 RGB images (CIFAR-10)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 Convolution
    branch1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 3x3 Convolution
    branch2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)

    # Branch 3: 5x5 Convolution
    branch3 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(input_layer)

    # Branch 4: 3x3 Max Pooling
    branch4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenating all branches
    concatenated = layers.concatenate([branch1, branch2, branch3, branch4])

    # Block 2: Global Average Pooling
    global_avg_pool = layers.GlobalAveragePooling2D()(concatenated)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(global_avg_pool)
    dense2 = layers.Dense(10)(dense1)  # Output layer has 10 classes for CIFAR-10

    # Generating weights and reshaping
    weights = layers.Dense(concatenated.shape[-1], activation='sigmoid')(dense1)
    weights_reshaped = layers.Reshape((1, 1, concatenated.shape[-1]))(weights)

    # Element-wise multiplication
    multiplied = layers.multiply([concatenated, weights_reshaped])

    # Final Global Average Pooling after multiplication
    final_avg_pool = layers.GlobalAveragePooling2D()(multiplied)

    # Output Layer
    output_layer = layers.Dense(10, activation='softmax')(final_avg_pool)

    # Constructing the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()