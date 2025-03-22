import keras
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block with a main and branch path
    def first_block(input_tensor):
        # Main path
        main_path_conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        main_path_dropout = Dropout(rate=0.3)(main_path_conv1)
        main_path_conv2 = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_path_dropout)
        
        # Branch path (identity)
        branch_path = input_tensor
        
        # Add both paths
        block_output = Add()([main_path_conv2, branch_path])
        
        return block_output
    
    # Second block with split and separable convolutions
    def second_block(input_tensor):
        def split(x):
            return tf.split(x, num_or_size_splits=3, axis=-1)
        
        split_layer = Lambda(split)(input_tensor)
        
        # Separable convolutions with different kernel sizes
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
        path1 = Dropout(rate=0.3)(path1)
        
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
        path2 = Dropout(rate=0.3)(path2)
        
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
        path3 = Dropout(rate=0.3)(path3)
        
        # Concatenate paths
        block_output = Concatenate()([path1, path2, path3])
        
        return block_output
    
    # Apply the first and second blocks
    block1_output = first_block(input_layer)
    block2_output = second_block(block1_output)
    
    # Final layers for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model