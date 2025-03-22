# Import necessary packages
from tensorflow.keras.layers import Input, concatenate, MaxPooling2D, GlobalAveragePooling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
import tensorflow as tf

def dl_model():
    """
    Function to create a deep learning model for image classification using MNIST dataset.
    
    The model consists of two specific blocks: 
    1. Block 1: Contains three parallel paths, each passing through max pooling layers of three different scales. 
               The pooling results from each path are then flattened into one-dimensional vectors and regularized using dropout layers. 
               These vectors are concatenated to form the output.
    2. Block 2: Starts with four parallel paths from the same input layer, each employing different convolution and pooling strategies to extract multi-scale features.
               The outputs of all paths are concatenated along the channel dimension. 
               After the above processing, the final classification results are output through two fully connected layers.
    
    Returns:
        model (tf.keras.Model): The constructed deep learning model.
    """
    
    # Define input layer with shape (28, 28, 1) representing MNIST dataset
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Three parallel paths with max pooling layers of three different scales
    path1 = MaxPooling2D(pool_size=(1, 1), strides=1)(input_layer)  # Scale 1
    path1 = MaxPooling2D(pool_size=(2, 2), strides=2)(path1)        # Scale 2
    path1 = MaxPooling2D(pool_size=(4, 4), strides=4)(path1)        # Scale 3
    
    path2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2)(path2)
    path2 = MaxPooling2D(pool_size=(4, 4), strides=4)(path2)
    
    path3 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(input_layer)
    path3 = MaxPooling2D(pool_size=(2, 2), strides=2)(path3)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=4)(path3)
    
    # Concatenate pooling results from each path
    output_block1 = concatenate([path1, path2, path3], axis=3)
    
    # Flatten and regularize output using dropout layer
    output_block1 = tf.keras.layers.Flatten()(output_block1)
    output_block1 = Dropout(0.2)(output_block1)
    
    # Reshape and add fully connected layer for transformation
    reshaped_output = Reshape((6 * 6, 3))(output_block1)
    output_block1 = Dense(128, activation='relu')(reshaped_output)
    
    # Block 2: Four parallel paths with different convolution and pooling strategies
    path1_block2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path1_block2 = MaxPooling2D(pool_size=(2, 2), strides=2)(path1_block2)
    
    path2_block2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path2_block2 = Conv2D(32, (1, 7), activation='relu')(path2_block2)
    path2_block2 = Conv2D(32, (7, 1), activation='relu')(path2_block2)
    path2_block2 = MaxPooling2D(pool_size=(2, 2), strides=2)(path2_block2)
    
    path3_block2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path3_block2 = Conv2D(32, (7, 1), activation='relu')(path3_block2)
    path3_block2 = Conv2D(32, (1, 7), activation='relu')(path3_block2)
    path3_block2 = Conv2D(32, (7, 1), activation='relu')(path3_block2)
    path3_block2 = MaxPooling2D(pool_size=(2, 2), strides=2)(path3_block2)
    
    path4_block2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path4_block2 = AveragePooling2D(pool_size=(2, 2), strides=2)(path4_block2)
    
    # Concatenate outputs of all paths along channel dimension
    output_block2 = concatenate([path1_block2, path2_block2, path3_block2, path4_block2], axis=3)
    
    # Flatten and add fully connected layers for classification
    output_block2 = tf.keras.layers.Flatten()(output_block2)
    output_block2 = Dense(128, activation='relu')(output_block2)
    output_block2 = Dropout(0.2)(output_block2)
    output_block2 = Dense(10, activation='softmax')(output_block2)
    
    # Create model by combining Block 1 and Block 2
    model = Model(inputs=input_layer, outputs=output_block2)
    
    return model

from tensorflow.keras.layers import Conv2D