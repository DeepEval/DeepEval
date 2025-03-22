import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups based on channels
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Main path with depthwise separable convolutions
    path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split[2])
    
    # Concatenate outputs from depthwise convolutions
    main_path_output = Concatenate()([path1, path2, path3])
    
    # Branch path with 1x1 convolution
    branch_path_output = Conv2D(filters=main_path_output.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs of the main path and branch path
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model