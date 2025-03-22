import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Lambda, Concatenate, Add, Dense, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path: Split the input along the channel dimension into 3 groups
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split_layer = Lambda(split_channels)(input_layer)
    
    # Apply depthwise separable convolution on each split
    conv1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    conv5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    
    # Concatenate the results from the three convolutions
    main_path_output = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Branch Path: A 1x1 convolution to align the number of output channels
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs from the main path and branch path
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model