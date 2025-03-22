import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras import backend as K
import tensorflow as tf

def dl_model():
    
    # Define the input layer with shape (32, 32, 3) for CIFAR-10 dataset
    input_layer = Input(shape=(32, 32, 3))
    
    # Use Lambda layer to split the input into three groups along the last dimension
    def split_input(input_tensor):
        x1, x2, x3 = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        return x1, x2, x3
    
    split_input_layer = Lambda(split_input)(input_layer)
    
    # Define the feature extraction block for each group
    def feature_extraction_block(input_tensor, kernel_size):
        return DepthwiseConv2D(kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
    
    # Apply the feature extraction block for each group with different kernel sizes
    x1 = feature_extraction_block(split_input_layer[0], 1)
    x2 = feature_extraction_block(split_input_layer[1], 3)
    x3 = feature_extraction_block(split_input_layer[2], 5)
    
    # Concatenate the outputs of the three groups
    output_tensor = Concatenate()([x1, x2, x3])
    
    # Add batch normalization layer
    bath_norm = BatchNormalization()(output_tensor)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Apply a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model