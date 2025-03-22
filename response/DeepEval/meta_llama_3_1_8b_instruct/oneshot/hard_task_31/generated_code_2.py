import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Add, Lambda, Reshape
from keras import backend as K
from keras import regularizers
import tensorflow as tf

def dl_model():
    
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the input layer
    input_layer = Input(shape=input_shape)
    
    # Block 1: Main path and branch path
    def block1(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout = Dropout(0.2)(conv)
        restore_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout)
        main_output = restore_conv
        
        # Branch path
        branch_output = input_tensor
        
        # Add the outputs from both paths
        output_tensor = Add()([main_output, branch_output])
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Block 2: Separable convolutional layers
    def block2(input_tensor):
        # Split the input into three groups along the last dimension
        def split(x):
            return tf.split(x, num_or_size_splits=3, axis=-1)
        
        split_layer = Lambda(split)(input_tensor)
        
        # Use separable convolutional layers of varying kernel sizes
        conv1 = tf.keras.layers.SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same')(split_layer[0])
        conv1 = Dropout(0.2)(conv1)
        
        conv2 = tf.keras.layers.SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(split_layer[1])
        conv2 = Dropout(0.2)(conv2)
        
        conv3 = tf.keras.layers.SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same')(split_layer[2])
        conv3 = Dropout(0.2)(conv3)
        
        # Concatenate the outputs from the three groups
        output_tensor = tf.concat([conv1, conv2, conv3], axis=-1)
        
        return output_tensor
    
    block2_output = block2(block1_output)
    
    # Flatten the output
    flatten_layer = Flatten()(block2_output)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model