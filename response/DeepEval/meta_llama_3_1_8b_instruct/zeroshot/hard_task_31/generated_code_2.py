# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, Add, Dropout, Lambda, concatenate, SeparableConv2D, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model using the Functional APIs of Keras for image classification on the CIFAR-10 dataset.
    
    The model consists of two main blocks:
    1. The first block features both a main path and a branch path. The main path begins with a <convolution, dropout> block to expand the width of the feature map, 
       followed by a convolutional layer to restore the number of channels to same as those of input. In parallel, the branch path directly connects to the input.
       The outputs from both paths are then added to produce the output of this block.
    2. The second block splits the input into three groups along the last dimension by encapsulating tf.split within Lambda layer, 
       with each group using separable convolutional layers of varying kernel sizes (1x1, 3x3, and 5x5) to extract features. Each convolution is followed by a dropout layer to mitigate overfitting.
       The outputs from the three groups are concatenated to create a unified feature representation.
    
    After processing through these two blocks to extract features, the model outputs the final predictions via a flattening layer and a fully connected layer.
    """
    
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Main path of the first block
    x = Conv2D(64, (3, 3), activation='relu')(inputs)  # Convolutional layer with 64 filters of size 3x3
    x = Dropout(0.2)(x)  # Dropout layer with 20% dropout rate
    x = Conv2D(64, (3, 3), activation='relu')(x)  # Convolutional layer with 64 filters of size 3x3
    
    # Branch path of the first block
    x_branch = inputs
    
    # Add the outputs of the main and branch paths
    x = Add()([x, x_branch])
    
    # Lambda layer to split the input into three groups along the last dimension
    x_split = Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    
    # First group
    x_1 = SeparableConv2D(32, (1, 1), activation='relu')(x_split[0])  # Separable convolutional layer with 32 filters of size 1x1
    x_1 = Dropout(0.2)(x_1)
    
    # Second group
    x_2 = SeparableConv2D(32, (3, 3), activation='relu')(x_split[1])  # Separable convolutional layer with 32 filters of size 3x3
    x_2 = Dropout(0.2)(x_2)
    
    # Third group
    x_3 = SeparableConv2D(32, (5, 5), activation='relu')(x_split[2])  # Separable convolutional layer with 32 filters of size 5x5
    x_3 = Dropout(0.2)(x_3)
    
    # Concatenate the outputs of the three groups
    x = concatenate([x_1, x_2, x_3], axis=-1)
    
    # Flatten the output
    x = Flatten()(x)
    
    # Output layer
    outputs = Dense(10, activation='softmax')(x)  # Fully connected layer with 10 units and softmax activation function
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model