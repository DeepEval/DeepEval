import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():     
    """
    A deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path and a branch path. The main path splits the input 
    into three groups by encapsulating tf.split within Lambda layer, each undergoing 
    feature extraction with convolutional layers of different kernel sizes (1x1, 3x3, and 5x5). 
    The outputs from these three groups are then concatenated. The branch path processes 
    the input with a 1x1 convolutional layer to align the number of output channels with 
    those of the main path. The outputs of the main and branch paths are combined through 
    addition to create fused features. Finally, the model performs classification using two 
    fully connected layers.
    """
    
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Main path
    def split_input(input_tensor):
        """
        Splits the input into three groups with convolutional layers of different kernel sizes.
        
        Args:
            input_tensor: The input to the main path.
        
        Returns:
            A concatenated tensor of the outputs from the three groups.
        """
        group1 = layers.Conv2D(32, (1, 1), activation='relu')(input_tensor)
        group2 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
        group3 = layers.Conv2D(32, (5, 5), activation='relu')(input_tensor)
        
        return layers.Concatenate()([group1, group2, group3])
    
    main_output = split_input(input_layer)
    
    # Branch path
    branch_output = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # Combine the main and branch paths
    fused_features = layers.Add()([main_output, branch_output])
    
    # Flatten and classification
    flatten_layer = layers.Flatten()(fused_features)
    dense1 = layers.Dense(128, activation='relu')(flatten_layer)
    output_layer = layers.Dense(10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model