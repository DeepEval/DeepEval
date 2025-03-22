import tensorflow as tf
from tensorflow.keras import layers, models, Input

def dl_model():
    # Define the input shape based on the MNIST dataset which is 28x28 pixels with 1 channel (grayscale)
    input_shape = (28, 28, 1)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 1x1 Convolutional Layer to increase dimensionality
    x = layers.Conv2D(32, (1, 1), strides=2, activation='relu')(inputs)
    
    # 3x3 Depthwise Separable Convolutional Layer for feature extraction
    x = layers.SeparableConv2D(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    
    # Another 1x1 Convolutional Layer to reduce dimensionality
    x = layers.Conv2D(32, (1, 1), strides=2, activation='relu')(x)
    
    # Flatten layer to convert the 2D matrix to a vector
    x = layers.Flatten()(x)
    
    # Fully Connected Layer for output
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()