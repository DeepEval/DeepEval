import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense
from keras import backend as K
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along channel
    split_input = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    
    # Define a depthwise separable convolutional layer with kernel size 1x1
    path1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
    
    # Define a depthwise separable convolutional layer with kernel size 3x3
    path2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input[1])
    
    # Define a depthwise separable convolutional layer with kernel size 5x5
    path3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_input[2])
    
    # Concatenate the outputs of the three paths
    main_path_output = Concatenate()([path1, path2, path3])
    
    # Define a branch path with 1x1 convolutional layer
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs from the main and branch paths
    add_output = Add()([main_path_output, branch_path])
    
    # Apply batch normalization
    bath_norm = BatchNormalization()(add_output)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Define the second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model