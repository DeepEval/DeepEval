import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def build_branch(input_tensor):
    # Block: Global Average Pooling
    x = layers.GlobalAveragePooling2D()(input_tensor)
    
    # Two fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # Weights with dimension equal to the number of channels in input
    channel_dim = input_tensor.shape[-1]
    weights = layers.Dense(channel_dim, activation='sigmoid')(x)

    # Reshape weights to match the input shape
    reshaped_weights = layers.Reshape((1, 1, channel_dim))(weights)

    # Element-wise multiplication with the block's input
    output = layers.multiply([input_tensor, reshaped_weights])

    return output

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Create two branches using the same block
    branch1 = build_branch(inputs)
    branch2 = build_branch(inputs)

    # Concatenate the outputs from both branches
    concatenated = layers.concatenate([branch1, branch2])

    # Flatten the concatenated outputs
    flattened = layers.Flatten()(concatenated)

    # Fully connected layer for classification
    output = layers.Dense(10, activation='softmax')(flattened)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=inputs, outputs=output)

    return model

# Example usage
model = dl_model()
model.summary()  # Print the model summary to verify the architecture