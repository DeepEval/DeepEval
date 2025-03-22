import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Split the input into three groups along the last dimension
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # First group remains unchanged
        group1 = split_layer[0]

        # Second group with a 3x3 convolution
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
        
        # Combine the output of the second group with the third group
        combined_group = Concatenate()([group2, split_layer[2]])
        
        # Additional 3x3 convolution
        group3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)

        # Concatenate the outputs of all three groups
        main_output = Concatenate()([group1, group2, group3])
        return main_output
    
    # Branch path
    def branch_path(input_tensor):
        branch_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_output
    
    # Get the outputs from both paths
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    
    # Fuse the outputs from both paths through addition
    combined_output = Add()([main_output, branch_output])
    
    # Final classification result
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model