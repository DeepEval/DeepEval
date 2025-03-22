import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Lambda, Concatenate, Conv2D, DepthwiseConv2D
from keras import backend as K
from keras import regularizers
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    max_pooling1 = Flatten()(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    max_pooling2 = Flatten()(max_pooling2)
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    max_pooling3 = Flatten()(max_pooling3)
    
    concatenated_output = Concatenate()([max_pooling1, max_pooling2, max_pooling3])
    
    dropout = Dropout(0.2)(concatenated_output)
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(dropout)
    
    # Reshape layer to prepare for second block
    reshaped_layer = Reshape((128,))(dense1)
    
    # Second block
    def split(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=4, axis=-1)
    
    split_layer = Lambda(split)(reshaped_layer)
    
    conv_output1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv_output1 = Reshape((128,))(conv_output1)
    
    conv_output2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv_output2 = Reshape((128,))(conv_output2)
    
    conv_output3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    conv_output3 = Reshape((128,))(conv_output3)
    
    conv_output4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split_layer[3])
    conv_output4 = Reshape((128,))(conv_output4)
    
    concatenated_conv_output = Concatenate()([conv_output1, conv_output2, conv_output3, conv_output4])
    
    flattened_output = Flatten()(concatenated_conv_output)
    
    output_layer = Dense(units=10, activation='softmax')(flattened_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model