import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():     
    # Construct the input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Define the main path
    def main_path(input_tensor):
        # Split the input into three groups along the last dimension
        group1, group2, group3 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # The first group remains unchanged
        output_group1 = group1
        
        # The second group undergoes feature extraction via a 3x3 convolutional layer
        output_group2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
        
        # The output of the second group is then combined with the third group
        combined_group = layers.Concatenate()([group3, output_group2])
        
        # An additional 3x3 convolution is applied to the combined group
        output_group3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)
        
        # The outputs of all three groups are concatenated to form the output of the main path
        output_tensor = layers.Concatenate()([output_group1, output_group2, output_group3])
        
        return output_tensor
    
    # Define the branch path
    def branch_path(input_tensor):
        # A 1x1 convolutional layer is applied to process the input
        output_tensor = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        return output_tensor
    
    # Construct the model
    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)
    
    # The outputs from both the main and branch paths are fused together through addition
    combined_output = layers.Add()([main_path_output, branch_path_output])
    
    # Flatten the combined output
    flattened_output = layers.Flatten()(combined_output)
    
    # A fully connected layer is applied to obtain the final classification result
    output_layer = layers.Dense(units=10, activation='softmax')(flattened_output)
    
    # Construct the final model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model