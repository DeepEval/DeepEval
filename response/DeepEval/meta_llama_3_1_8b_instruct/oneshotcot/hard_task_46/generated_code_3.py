import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Deep learning model for image classification using the CIFAR-10 dataset.
    
    The model comprises two main blocks. In the first block, the input is splitted into three groups along the channel 
    by encapsulating tf.split within Lambda layer, each group uses separable convolutional with different kernel sizes 
    (1x1, 3x3, and 5x5) to extract features. The outputs from these three groups are then concatenated. In the second 
    block, the block features multiple branches for enhanced feature extraction. The input is processed through: 
    1.a 3x3 convolution, 2.a series of layers consisting of a 1x1 convolution followed by two 3x3 convolutions, 
    3.a max pooling branch. After feature extraction, the outputs from all branches are concatenated for further 
    integration. After processing through both blocks, the concatenated outputs undergo global average pooling, 
    followed by a fully connected layer that produces the final classification results.
    """
    
    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Define the first block
    def block1(input_tensor):
        # Split the input into three groups along the channel
        group1, group2, group3 = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        
        # Apply separable convolutional with different kernel sizes to each group
        conv1 = layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(group1)
        conv2 = layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu')(group2)
        conv3 = layers.SeparableConv2D(32, kernel_size=(5, 5), activation='relu')(group3)
        
        # Concatenate the outputs from the three groups
        output_tensor = layers.Concatenate()([conv1, conv2, conv3])
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Define the second block
    def block2(input_tensor):
        # Process through multiple branches for enhanced feature extraction
        path1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(input_tensor)
        path2 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(path2)
        path2 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(path2)
        path3 = layers.MaxPooling2D(pool_size=(2, 2))(input_tensor)
        
        # Concatenate the outputs from all branches
        output_tensor = layers.Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    block2_output = block2(block1_output)
    
    # Apply global average pooling
    gap_output = layers.GlobalAveragePooling2D()(block2_output)
    
    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(gap_output)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model