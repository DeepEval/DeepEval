import keras
from keras.layers import Input, Conv2D, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block: Dual-path structure
    def dual_path_block(input_tensor):
        # Main path
        main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
        main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(main_path)
        
        # Branch path (bypass)
        branch_path = input_tensor
        
        # Combine both paths using addition
        output_tensor = Add()([main_path, branch_path])
        return output_tensor
    
    # Apply first block
    block1_output = dual_path_block(input_layer)
    
    # Second Block: Split and depthwise separable convolutions
    def split_and_depthwise_block(input_tensor):
        # Split input into 3 groups along the channel dimension
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply depthwise separable convolutions with different kernel sizes
        group1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(groups[0])
        group2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(groups[1])
        group3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(groups[2])
        
        # Concatenate outputs
        output_tensor = Concatenate()([group1, group2, group3])
        return output_tensor
    
    # Apply second block
    block2_output = split_and_depthwise_block(block1_output)
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model