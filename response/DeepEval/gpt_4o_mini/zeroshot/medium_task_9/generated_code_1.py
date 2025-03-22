import tensorflow as tf
from tensorflow.keras import layers, models

def basic_block(x, filters):
    """Defines a basic block consisting of a convolution, batch normalization, and ReLU."""
    # Main path
    shortcut = x
    
    # Convolutional layer
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Branch for feature extraction
    branch = layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(shortcut)
    
    # Feature fusion
    x = layers.add([x, branch])
    x = layers.ReLU()(x)
    
    return x

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to reduce dimensionality
    x = layers.Conv2D(16, kernel_size=(3, 3), padding='same')(inputs)
    
    # Two consecutive basic blocks
    x = basic_block(x, 16)
    x = basic_block(x, 16)
    
    # Average pooling layer to downsample
    x = layers.AveragePooling2D(pool_size=(8, 8))(x)
    
    # Flatten the feature map
    x = layers.Flatten()(x)
    
    # Fully connected layer for output
    x = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=x)
    
    return model

# Example of how to create the model
model = dl_model()
model.summary()