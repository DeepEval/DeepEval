import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda, concatenate
from keras import backend as K
from keras.models import Model
from keras import regularizers
from tensorflow.keras.layers import SeparableConv2D
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First block with main and branch paths
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Dropout(0.2)(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    branch_path = input_layer
    output_block = concatenate([main_path, branch_path])
    
    # Second block with separable convolutions and dropout
    def separable_convolution(input_tensor, kernel_size):
        return SeparableConv2D(filters=64, kernel_size=kernel_size, activation='relu')(input_tensor)
    
    separable_conv1 = separable_convolution(output_block, (1, 1))
    separable_conv2 = separable_convolution(output_block, (3, 3))
    separable_conv3 = separable_convolution(output_block, (5, 5))
    separable_conv_output = concatenate([separable_conv1, separable_conv2, separable_conv3])
    
    # Dropouts for regularization
    separable_conv_output = Dropout(0.2)(separable_conv_output)
    
    # Split the input into three groups along the last dimension
    split_output = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(separable_conv_output)
    
    # Extract features using separable convolutions
    features1 = separable_convolution(split_output[0], (1, 1))
    features2 = separable_convolution(split_output[1], (3, 3))
    features3 = separable_convolution(split_output[2], (5, 5))
    
    # Concatenate the feature maps from the three groups
    output_block = concatenate([features1, features2, features3])
    
    # Flatten the output
    flatten_layer = Flatten()(output_block)
    
    # Dense layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model