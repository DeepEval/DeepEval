import keras
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    # Main path: <convolution, dropout> -> convolution
    main_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_dropout = Dropout(rate=0.3)(main_conv1)
    main_conv2 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_dropout)
    
    # Branch path: direct connection
    branch_path = input_layer
    
    # Combine the paths
    first_block_output = Add()([main_conv2, branch_path])
    
    # Second Block
    def split_and_process(input_tensor):
        # Split into three along last dimension (channels)
        splits = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        
        # Path 1: SeparableConv2D 1x1
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        path1 = Dropout(rate=0.3)(path1)
        
        # Path 2: SeparableConv2D 3x3
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        path2 = Dropout(rate=0.3)(path2)
        
        # Path 3: SeparableConv2D 5x5
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        path3 = Dropout(rate=0.3)(path3)
        
        # Concatenate the paths
        return Concatenate()([path1, path2, path3])
    
    second_block_output = Lambda(split_and_process)(first_block_output)
    
    # Final Layers
    flatten_layer = Flatten()(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model