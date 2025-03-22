import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input Layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution and 3x3 convolution
    branch_1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch_1)
    branch_1 = layers.Dropout(0.5)(branch_1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
    branch_2 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch_2 = layers.Conv2D(64, (1, 7), activation='relu', padding='same')(branch_2)
    branch_2 = layers.Conv2D(64, (7, 1), activation='relu', padding='same')(branch_2)
    branch_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch_2)
    branch_2 = layers.Dropout(0.5)(branch_2)

    # Branch 3: Max Pooling
    branch_3 = layers.MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch_3 = layers.Dropout(0.5)(branch_3)

    # Concatenate the outputs from all branches
    concatenated = layers.Concatenate()([branch_1, branch_2, branch_3])

    # Fully connected layers
    x = layers.Flatten()(concatenated)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()