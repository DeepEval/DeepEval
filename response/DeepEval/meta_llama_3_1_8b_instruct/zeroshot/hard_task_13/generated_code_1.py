# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of two main blocks:
    1. The first block processes the input through four parallel branches: a 1x1 convolution, a 3x3 convolution, a 5x5 convolution, and a 3x3 max pooling operation.
    2. The second block reduces the dimensionality of the feature maps using global average pooling and then passes the pooled features through two fully connected layers.
    
    Finally, the output of the fully connected layers is reshaped to match the input's shape, multiplied element-wise with the input feature map, and then processed by a final fully connected layer to produce the output probability distribution.
    """
    
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Define the four parallel branches in the first block
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch3 = layers.Conv2D(32, (5, 5), activation='relu')(inputs)
    branch4 = layers.MaxPooling2D((3, 3), strides=(2, 2))(inputs)
    
    # Concatenate the outputs of the four parallel branches
    merged = layers.Concatenate()([branch1, branch2, branch3, branch4])
    
    # Define the second block
    x = layers.Conv2D(32, (1, 1))(merged)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Reshape the output of the fully connected layers to match the input's shape
    x = layers.Reshape((32, 32, 32))(x)
    
    # Multiply the reshaped output with the input feature map
    x = layers.Multiply()([x, inputs])
    
    # Define the final fully connected layer
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Define the model
model = dl_model()
model.summary()