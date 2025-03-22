# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path and a branch path. In the main path, the input is split into three groups 
    along the channel by encapsulating tf.split within Lambda layer, each undergoing multi-scale feature extraction 
    with separable convolutional layers of varying kernel sizes (1x1, 3x3, and 5x5). The outputs from these groups 
    are concatenated to produce the output of the main path. The branch path applies a 1x1 convolutional layer to the 
    input to align the number of output channels with those of the main path. The outputs from both paths are then fused 
    through addition. Finally, the combined output is flattened into a one-dimensional vector and passed through two fully 
    connected layers for a 10-class classification task.
    
    Parameters:
    None
    
    Returns:
    model (keras.Model): The constructed deep learning model.
    """

    # Define the input shape of the model, which is a 32x32 color image
    input_shape = (32, 32, 3)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Split the input into three groups along the channel using Lambda layer
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Define the main path
    main_path = layers.Concatenate()([
        # Apply separable convolutional layers of varying kernel sizes (1x1, 3x3, and 5x5)
        layers.SeparableConv2D(32, (1, 1), activation='relu', name='conv1')(split_layer[0]),
        layers.SeparableConv2D(32, (3, 3), activation='relu', name='conv2')(split_layer[1]),
        layers.SeparableConv2D(32, (5, 5), activation='relu', name='conv3')(split_layer[2])
    ])
    
    # Apply average pooling to downsample the feature maps
    avg_pool = layers.GlobalAveragePooling2D()(main_path)
    
    # Define the branch path
    branch_path = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Fuse the outputs from both paths through addition
    fused_output = layers.Add()([main_path, branch_path])
    
    # Apply average pooling to downsample the feature maps
    avg_pool_branch = layers.GlobalAveragePooling2D()(branch_path)
    
    # Fuse the outputs from both paths through addition
    fused_output = layers.Add()([fused_output, avg_pool_branch])
    
    # Flatten the output into a one-dimensional vector
    flatten_layer = layers.Flatten()(fused_output)
    
    # Apply two fully connected layers for a 10-class classification task
    dense_layer1 = layers.Dense(128, activation='relu')(flatten_layer)
    outputs = layers.Dense(10, activation='softmax')(dense_layer1)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model