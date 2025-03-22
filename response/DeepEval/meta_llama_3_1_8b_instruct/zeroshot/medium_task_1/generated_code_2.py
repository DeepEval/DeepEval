# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model using the Keras Functional API.
    The model is designed for image classification using the CIFAR-10 dataset.
    
    Model Architecture:
    - Conv2D (32 filters, kernel size 3, activation='relu')
    - Conv2D (64 filters, kernel size 3, activation='relu')
    - MaxPooling2D (pool size 2)
    - Concatenate (input and feature map)
    - Flatten
    - Dense (64 units, activation='relu')
    - Dense (10 units, activation='softmax')
    
    :return: A compiled Keras model.
    """
    
    # Define the input layer with shape (32, 32, 3) for CIFAR-10 images
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Conv2D layer with 32 filters, kernel size 3, and ReLU activation
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    
    # Conv2D layer with 64 filters, kernel size 3, and ReLU activation
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(conv1)
    
    # MaxPooling2D layer with pool size 2
    pool = layers.MaxPooling2D((2, 2))(conv2)
    
    # Concatenate the input layer and the feature map from the max-pooling layer
    concatenated = layers.Concatenate()([input_layer, pool])
    
    # Flatten the concatenated feature map
    flattened = layers.Flatten()(concatenated)
    
    # Dense layer with 64 units and ReLU activation
    dense1 = layers.Dense(64, activation='relu')(flattened)
    
    # Output layer with 10 units and softmax activation for probability distribution
    output_layer = layers.Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Usage:
model = dl_model()
model.summary()