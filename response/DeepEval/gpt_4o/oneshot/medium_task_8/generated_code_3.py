import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    def main_path(input_tensor):
        # Splitting the input tensor into three groups along the last dimension
        split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # First group remains unchanged
        group1 = split_layers[0]
        
        # Second group undergoes feature extraction via a 3x3 convolution
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layers[1])
        
        # Combine second group output with the third group and pass through an additional 3x3 convolution
        combined_group = Concatenate()([group2, split_layers[2]])
        group3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)
        
        # Concatenate all three groups to form the output of the main path
        main_path_output = Concatenate()([group1, group2, group3])
        
        return main_path_output
    
    # Branch Path
    def branch_path(input_tensor):
        # Process the input with a 1x1 convolutional layer
        branch_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_output
    
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    
    # Fuse outputs from both paths via addition
    fused_output = Add()([main_output, branch_output])
    
    # Flatten the combined output and pass through a fully connected layer for final classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model