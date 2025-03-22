import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Add input layer
    input_layer = keras.Input(shape=input_shape)

    # First block
    def block1(input_tensor):
        # Split the input into three groups along the last dimension
        group1, group2, group3 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        # Apply depthwise separable convolutional layers with varying kernel sizes (1x1, 3x3, and 5x5) to extract features
        path1 = layers.SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(group1)
        path2 = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(group2)
        path3 = layers.SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(group3)
        
        # Concatenate the outputs from these groups
        output_tensor = layers.Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    # Apply the first block
    block1_output = block1(input_layer)
    
    # Second block
    def block2(input_tensor):
        # Define multiple branches for feature extraction
        branch1 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        branch3 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
        branch4 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        branch4 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4)
        
        # Concatenate the outputs from all branches
        output_tensor = layers.Concatenate()([branch1, branch2, branch3, branch4])
        
        return output_tensor
    
    # Apply the second block
    block2_output = block2(block1_output)
    
    # Apply batch normalization
    batch_norm = layers.BatchNormalization()(block2_output)
    
    # Apply flattening layer
    flatten_layer = layers.Flatten()(batch_norm)
    
    # Apply fully connected layer
    output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model