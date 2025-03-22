import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images of shape (32, 32, 3)
    input_layer = layers.Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    x = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Branch 1: 3x3 convolution for local feature extraction
    branch1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Branch 2: Max pooling followed by 3x3 convolution and upsampling
    branch2 = layers.MaxPooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Max pooling followed by 3x3 convolution and upsampling
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.UpSampling2D(size=(2, 2))(branch3)

    # Concatenate all branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # Final 1x1 convolutional layer to reduce the feature maps
    x = layers.Conv2D(128, (1, 1), activation='relu')(concatenated)

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    output_layer = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()