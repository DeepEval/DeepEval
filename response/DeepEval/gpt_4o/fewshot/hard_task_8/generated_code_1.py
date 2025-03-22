import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Main and Branch Paths
    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    main_conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_depthwise_conv)
    
    # Branch path
    branch_depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_depthwise_conv)
    
    # Concatenate outputs from main and branch paths
    concat = Concatenate(axis=-1)([main_conv2, branch_conv])
    
    # Block 2: Reshape and Channel Shuffle
    # Get shape of concatenated features and reshape for channel shuffling
    height, width, channels = concat.shape[1], concat.shape[2], concat.shape[3]
    groups = 4
    channels_per_group = channels // groups
    
    reshaped = Reshape(target_shape=(height, width, groups, channels_per_group))(concat)
    permuted = Permute((1, 2, 4, 3))(reshaped)
    reshuffled = Reshape(target_shape=(height, width, channels))(permuted)
    
    # Final classification layer
    flatten = Flatten()(reshuffled)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model