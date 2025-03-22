import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Initial 1x1 Convolution
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # First Branch (Local Feature Extraction)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)

    # Second Branch (Average Pooling followed by Convolution and Upsampling)
    branch2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(branch2)

    # Third Branch (Average Pooling followed by Convolution and Upsampling)
    branch3 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(branch3)

    # Concatenate the outputs of all branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # Refine with a 1x1 convolution
    x = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(concatenated)

    # Flatten and fully connected layer for classification
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()