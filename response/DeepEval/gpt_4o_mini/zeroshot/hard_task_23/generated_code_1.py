import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # First branch: Local feature extraction
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)

    # Second branch: Downsampling + Convolution + Upsampling
    branch2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(branch2)

    # Third branch: Downsampling + Convolution + Upsampling
    branch3 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(branch3)

    # Concatenate outputs from all branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # Final 1x1 convolutional layer to refine the features
    refined = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(concatenated)

    # Flatten the output for the fully connected layer
    flattened = layers.Flatten()(refined)

    # Fully connected layer for classification
    output_layer = layers.Dense(10, activation='softmax')(flattened)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # Display the model architecture