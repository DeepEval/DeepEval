import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Reshape, Activation, Flatten, Dense
from keras import backend as K
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def split_input(input_tensor):
        return Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
    
    split_layer = split_input(input_layer)
    
    # Group 1: 1x1 convolution
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[0])
    conv1 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same')(conv1)
    
    # Group 2: 3x3 convolution
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(split_layer[1])
    conv2 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same')(conv2)
    
    # Group 3: 5x5 convolution
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same')(split_layer[2])
    conv3 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same')(conv3)
    
    # Concatenate the outputs of the three groups
    output_tensor = Concatenate()([conv1, conv2, conv3])
    
    # Reshape the concatenated output
    reshape_layer = Reshape((-1,))(output_tensor)
    
    # Flatten the reshaped output
    flatten_layer = Flatten()(reshape_layer)
    
    # Fully connected layer for classification
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model