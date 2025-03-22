import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 64)
    inputs = layers.Input(shape=input_shape)
    
    # Main path
    # 1x1 convolution for dimensionality reduction
    x = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(inputs)
    
    # Parallel convolutional layers
    conv1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    
    # Concatenate outputs of parallel layers
    concatenated = layers.Concatenate()([conv1, conv2])
    
    # Branch path
    # 3x3 convolution to match dimensions
    branch = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    
    # Combine main and branch paths
    combined = layers.Add()([concatenated, branch])
    
    # Flatten and fully connected layers for classification
    x = layers.Flatten()(combined)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # assuming 10 classes for classification
    
    # Define the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
model = dl_model()
model.summary()