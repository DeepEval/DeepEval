# Import necessary packages from Keras and other libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import glorot_uniform
import numpy as np

def dl_model():
    """
    This function constructs a deep learning model using the Functional API of Keras.
    
    The model features two parallel branches, each consisting of a same block. 
    The block contains three sequential convolutional layers, with the outputs of each layer directed to separate main paths. 
    Additionally, a parallel branch connects the input through a convolutional layer. 
    The outputs from all paths are combined using an addition operation. 
    Subsequently, the outputs from the two blocks are concatenated. 
    After passing through a flattening layer and a fully connected layer, the model produces the final classification results.
    
    Returns:
    model: The constructed deep learning model.
    """
    
    # Define the input shape for the model
    inputs = keras.Input(shape=(28, 28, 1))
    
    # Define the same block, which consists of three sequential convolutional layers
    block1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    block1 = layers.Conv2D(32, (3, 3), activation='relu')(block1)
    block1 = layers.Conv2D(32, (3, 3), activation='relu')(block1)
    
    # Define the parallel branch that connects the input through a convolutional layer
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    
    # Combine the outputs from all paths using an addition operation
    combined = layers.Add()([block1, branch2])
    
    # Define the second same block
    block2 = layers.Conv2D(64, (3, 3), activation='relu')(combined)
    block2 = layers.Conv2D(64, (3, 3), activation='relu')(block2)
    block2 = layers.Conv2D(64, (3, 3), activation='relu')(block2)
    
    # Concatenate the outputs from the two blocks
    concatenated = layers.Concatenate()([block1, block2])
    
    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)
    
    # Define the fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flattened)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create an instance of the model
model = dl_model()

# Print a summary of the model
model.summary()