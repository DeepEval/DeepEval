import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    A deep learning model for image classification using the CIFAR-10 dataset.
    
    The model features a main path and a branch path. In the main path, the input is 
    split into three groups along the channel by encapsulating tf.split within Lambda 
    layer, each of which undergoes feature extraction via depthwise separable 
    convolutional layers with varying kernel sizes: 1x1, 3x3, and 5x5. The outputs from 
    these three groups are concatenated to produce the main path output. The branch path 
    employs a 1x1 convolutional layer to align the number of output channels with those 
    of the main path. Finally, the outputs from both the main and branch paths are added. 
    The model concludes with two fully connected layers for the 10-class classification task.
    """
    
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Split the input into three groups along the channel
    split_input = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    # Define the main path
    main_path = []
    for i in range(3):
        # Apply depthwise separable convolutional layer with varying kernel sizes
        x = layers.DepthwiseConv2D(kernel_size=(1, 1), activation='relu')(split_input[i])
        x = layers.Conv2D(kernel_size=(3, 3), activation='relu')(x)
        x = layers.DepthwiseConv2D(kernel_size=(5, 5), activation='relu')(x)
        
        # Append the output to the main path
        main_path.append(x)
    
    # Concatenate the outputs from the three groups
    main_output = layers.Concatenate()(main_path)
    
    # Define the branch path
    branch_output = layers.Conv2D(kernel_size=(1, 1), activation='relu')(inputs)
    
    # Add the outputs from the main and branch paths
    add_output = layers.Add()([main_output, branch_output])
    
    # Flatten the output
    flat_output = layers.Flatten()(add_output)
    
    # Apply two fully connected layers for the 10-class classification task
    outputs = layers.Dense(128, activation='relu')(flat_output)
    outputs = layers.Dense(10, activation='softmax')(outputs)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

model = dl_model()