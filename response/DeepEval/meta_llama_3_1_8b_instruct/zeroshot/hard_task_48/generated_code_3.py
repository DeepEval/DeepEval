# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function generates a deep learning model for image classification using Functional APIs of Keras.
    The input data set is CIFAR-10. The model consists of two blocks.
    
    Block 1 splits the input into three groups, each extracting features through separable convolutional layers 
    with different kernel sizes (1x1, 3x3, 5x5), and employs batch normalization to enhance model performance.
    The outputs of the three groups are then concatenated together.
    
    Block 2 includes four parallel branches, each processing the input through different layers of convolution, 
    pooling, and concatenation. Finally, the outputs of these four paths are concatenated to form a comprehensive 
    multi-channel feature map. After the above processing, the final classification result is output through a 
    flattening layer and a fully connected layer.
    
    Parameters:
    None
    
    Returns:
    model: A deep learning model for image classification.
    """
    
    # Define the input layer with shape (32, 32, 3) for CIFAR-10
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Block 1: Split the input into three groups and extract features
    block1 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Group 1: 1x1 separable convolutional layer with batch normalization
    group1 = layers.SeparableConv2D(32, (1, 1), activation='relu')(block1[0])
    group1 = layers.BatchNormalization()(group1)
    
    # Group 2: 3x3 separable convolutional layer with batch normalization
    group2 = layers.SeparableConv2D(32, (3, 3), activation='relu')(block1[1])
    group2 = layers.BatchNormalization()(group2)
    
    # Group 3: 5x5 separable convolutional layer with batch normalization
    group3 = layers.SeparableConv2D(32, (5, 5), activation='relu')(block1[2])
    group3 = layers.BatchNormalization()(group3)
    
    # Concatenate the outputs of the three groups
    block1 = layers.Concatenate()([group1, group2, group3])
    
    # Block 2: Four parallel branches with different layers of convolution, pooling, and concatenation
    block2 = layers.Conv2D(32, (1, 1), activation='relu')(block1)
    
    # Path 1: 1x1 convolution
    path1 = block2
    
    # Path 2: 3x3 average pooling followed by 1x1 convolution
    path2 = layers.AveragePooling2D((3, 3))(block1)
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(path2)
    
    # Path 3: 1x1 convolution, split into two sub-paths with 1x3 and 3x1 convolutions
    path3 = layers.Conv2D(32, (1, 1), activation='relu')(block1)
    path3_1 = layers.Conv2D(32, (1, 3), activation='relu')(path3)
    path3_2 = layers.Conv2D(32, (3, 1), activation='relu')(path3)
    path3 = layers.Concatenate()([path3_1, path3_2])
    
    # Path 4: 1x1 convolution, followed by 3x3 convolution, split into two sub-paths with 1x3 and 3x1 convolutions
    path4 = layers.Conv2D(32, (1, 1), activation='relu')(block1)
    path4 = layers.Conv2D(32, (3, 3), activation='relu')(path4)
    path4_1 = layers.Conv2D(32, (1, 3), activation='relu')(path4)
    path4_2 = layers.Conv2D(32, (3, 1), activation='relu')(path4)
    path4 = layers.Concatenate()([path4_1, path4_2])
    
    # Concatenate the outputs of the four paths
    block2 = layers.Concatenate()([path1, path2, path3, path4])
    
    # Flatten the output
    x = layers.Flatten()(block2)
    
    # Output layer with 10 units for CIFAR-10 classification
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model