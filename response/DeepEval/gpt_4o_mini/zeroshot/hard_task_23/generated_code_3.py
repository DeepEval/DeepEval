import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Initial 1x1 convolutional layer
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # Branch 1: Local feature extraction with 3x3 convolutions
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)

    # Branch 2: Average pooling followed by convolution and upsampling
    branch2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(branch2)  # Upsampling

    # Branch 3: Average pooling followed by convolution and upsampling
    branch3 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(branch3)  # Upsampling

    # Concatenate outputs from all branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # Final refinement with a 1x1 convolutional layer
    x = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(concatenated)

    # Flattening the output
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
model.summary()  # To display the model architecture