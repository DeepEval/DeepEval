import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():     
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model splits the input image into three channel groups and applies separable 
    convolutional layers of varying sizes (1x1, 3x3, and 5x5) on each group. The outputs 
    from these three groups are concatenated and then passed through three fully connected 
    layers to produce the final probability outputs.
    """
    
    input_layer = keras.Input(shape=(32, 32, 3))  # Define the input shape
    
    # Split the input image into three channel groups
    channel_groups = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define a block for feature extraction
    def block(input_tensor):
        path1 = layers.SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        path3 = layers.SeparableConv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_tensor)
        
        # Concatenate the outputs from the three paths
        output_tensor = layers.Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    # Apply the block to each channel group
    group1_output = block(channel_groups[0])
    group2_output = block(channel_groups[1])
    group3_output = block(channel_groups[2])
    
    # Concatenate the outputs from the three groups
    output_tensor = layers.Concatenate()([group1_output, group2_output, group3_output])
    
    # Apply batch normalization and flatten the output
    batch_norm = layers.BatchNormalization()(output_tensor)
    flatten_layer = layers.Flatten()(batch_norm)
    
    # Define the output layers
    dense1 = layers.Dense(units=128, activation='relu')(flatten_layer)
    dense2 = layers.Dense(units=64, activation='relu')(dense1)
    output_layer = layers.Dense(units=10, activation='softmax')(dense2)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model