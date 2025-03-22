import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

def dl_model():
    # Input shape for CIFAR-10 images (32, 32, 3)
    input_shape = (32, 32, 3)
    inputs = layers.Input(shape=input_shape)
    
    # Convolutional base to extract features from images
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Global Average Pooling to capture global information
    x = layers.GlobalAveragePooling2D()(x)
    
    # Two fully connected layers to generate weights
    dense1 = layers.Dense(128, activation='relu')(x)
    dense2 = layers.Dense(128, activation='relu')(dense1)
    
    # Generating the weights (channel-wise correlations)
    weights = layers.Dense(input_shape[2], activation='sigmoid')(dense2)
    
    # Reshaping weights to the input shape
    reshaped_weights = layers.Reshape((1, 1, input_shape[2]))(weights)

    # Multiply element-wise with input features
    scaled_input = layers.multiply([inputs, reshaped_weights])
    
    # Flatten the result for the final output layer
    x = layers.Flatten()(scaled_input)
    
    # Final output layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()