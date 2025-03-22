import keras
from keras.layers import Input, Lambda, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense
from keras import backend as K
import tensorflow as tf

def dl_model():
    # Input layer with shape 32x32x3
    input_layer = Input(shape=(32, 32, 3))
    
    # Lambda layer to split the input along the last dimension into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Define the main path
    def main_path(group):
        # Group 1 remains unchanged
        group1 = group
        
        # Group 2 undergoes feature extraction via a 3x3 convolutional layer
        group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group)
        
        # Combine the output of group 2 with group 3 before passing through an additional 3x3 convolution
        combined = Concatenate()([group1, group2])
        combined = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined)
        
        # Concatenate the outputs of all three groups to form the output of the main path
        output = Concatenate()([group1, group2, combined])
        
        return output
    
    # Split the input into three groups and apply the main path to each group
    group1, group2, group3 = split_layer
    main_output = main_path(group1)
    main_output = main_path(group2)
    main_output = main_path(group3)
    main_output = Concatenate()([main_output, main_output, main_output])
    
    # Branch path employs a 1x1 convolutional layer to process the input
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    
    # Fuse the outputs from both the main and branch paths through addition
    fused_output = Add()([main_output, branch_output])
    
    # Batch normalization
    bath_norm = BatchNormalization()(fused_output)
    
    # Flatten the combined output
    flatten_layer = Flatten()(bath_norm)
    
    # Output layer with softmax activation for 10-class classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model