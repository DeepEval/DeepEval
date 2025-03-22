import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Create the input layer
    input_layer = keras.Input(shape=input_shape)
    
    # Define the first block
    def block1(input_tensor):
        
        # Apply max pooling with different scales
        pool1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        pool1_flat = layers.Flatten()(pool1)
        
        pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        pool2_flat = layers.Flatten()(pool2)
        
        pool3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        pool3_flat = layers.Flatten()(pool3)
        
        # Concatenate the flattened outputs and apply dropout
        output_tensor = layers.Concatenate()([pool1_flat, pool2_flat, pool3_flat])
        output_tensor = layers.Dropout(0.2)(output_tensor)
        
        return output_tensor
    
    # Apply the first block
    block1_output = block1(input_layer)
    
    # Transform the output into a four-dimensional tensor
    dense1 = layers.Dense(128, activation='relu')(block1_output)
    reshape = layers.Reshape((4, 32))(dense1)
    
    # Define the second block
    def block2(input_tensor):
        
        # Split the input into four groups
        groups = layers.Lambda(lambda x: tf.split(x, 4, axis=-1))(input_tensor)
        
        # Apply separable convolution with different kernel sizes
        group1 = layers.SeparableConv2D(32, (1, 1), activation='relu', padding='same')(groups[0])
        group2 = layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(groups[1])
        group3 = layers.SeparableConv2D(32, (5, 5), activation='relu', padding='same')(groups[2])
        group4 = layers.SeparableConv2D(32, (7, 7), activation='relu', padding='same')(groups[3])
        
        # Concatenate the outputs from the four groups
        output_tensor = layers.Concatenate()([group1, group2, group3, group4])
        
        return output_tensor
    
    # Apply the second block
    block2_output = block2(reshape)
    
    # Flatten the output and apply a fully connected layer
    flatten = layers.Flatten()(block2_output)
    output_layer = layers.Dense(10, activation='softmax')(flatten)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model