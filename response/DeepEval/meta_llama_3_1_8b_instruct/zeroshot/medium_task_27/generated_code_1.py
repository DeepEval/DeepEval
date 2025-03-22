# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Convolutional branch 1: 3x3 kernel
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    
    # Convolutional branch 2: 5x5 kernel
    y = layers.Conv2D(32, 5, activation='relu')(inputs)
    y = layers.Conv2D(64, 5, activation='relu')(y)
    y = layers.MaxPooling2D(2, 2)(y)
    
    # Combine the outputs of the two branches through addition
    combined = layers.Add()([x, y])
    
    # Global average pooling layer to compress the features
    global_avg_pool = layers.GlobalAveragePooling2D()(combined)
    
    # Fully connected layer 1
    fc1 = layers.Dense(128, activation='relu')(global_avg_pool)
    
    # Fully connected layer 2 (softmax activation)
    outputs = layers.Dense(10, activation='softmax')(fc1)
    
    # Multiply the output of each branch by its corresponding weight and add to produce the final weighted output
    weighted_output = layers.Add()([layers.Lambda(lambda x: x * 0.7)(x), layers.Lambda(lambda x: x * 0.3)(y)])
    
    # Final fully connected layer for probability distribution
    final_outputs = layers.Dense(10, activation='softmax')(weighted_output)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=final_outputs)
    
    return model