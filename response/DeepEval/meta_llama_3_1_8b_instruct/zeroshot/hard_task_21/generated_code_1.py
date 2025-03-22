# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model features a main path and a branch path. In the main path, the input is split into three groups 
    along the channel by encapsulating tf.split within Lambda layer, each of which undergoes feature extraction 
    via depthwise separable convolutional layers with varying kernel sizes: 1x1, 3x3, and 5x5. The outputs from 
    these three groups are concatenated to produce the main path output. The branch path employs a 1x1 convolutional 
    layer to align the number of output channels with those of the main path. Finally, the outputs from both the main 
    and branch paths are added. The model concludes with two fully connected layers for the 10-class classification task.
    
    Returns:
        model (keras.Model): The constructed deep learning model.
    """
    
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Create the model
    inputs = keras.Input(shape=input_shape)
    
    # Main path
    main_path = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    main_path_1x1 = layers.SeparableConv2D(6, 1, activation='relu')(main_path[0])
    main_path_3x3 = layers.SeparableConv2D(6, (3, 3), activation='relu')(main_path[1])
    main_path_5x5 = layers.SeparableConv2D(6, (5, 5), activation='relu')(main_path[2])
    main_path_output = layers.Concatenate()([main_path_1x1, main_path_3x3, main_path_5x5])
    
    # Branch path
    branch_path = layers.Conv2D(6, 1, activation='relu')(inputs)
    
    # Add the main and branch paths
    output = layers.Add()([main_path_output, branch_path])
    
    # Flatten the output
    output = layers.Flatten()(output)
    
    # Dense layers for classification
    output = layers.Dense(128, activation='relu')(output)
    outputs = layers.Dense(10, activation='softmax')(output)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model