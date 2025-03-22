import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def basic_block(input_tensor, filters):
    # Main path
    x = layers.Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Branch path
    branch = input_tensor
    
    # Combine both paths
    x = layers.Add()([x, branch])
    x = layers.ReLU()(x)
    
    return x

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image shape
    inputs = layers.Input(shape=input_shape)
    
    # First convolution layer to adjust input dimensions
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    
    # First level: Basic block
    x = basic_block(x, 16)
    
    # Second level: Two residual blocks
    for _ in range(2):
        # Main path with basic block
        main_path = basic_block(x, 16)
        
        # Branch path with independent convolution
        branch = layers.Conv2D(16, (1, 1), padding='same')(x)
        
        # Combine both paths
        x = layers.Add()([main_path, branch])
    
    # Third level: Global branch with convolution
    global_branch = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    
    # Final output after second level residual structure
    x = layers.Add()([x, global_branch])
    
    # Average pooling and fully connected layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs, outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()  # Display the model architecture