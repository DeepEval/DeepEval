import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import he_normal
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def split_input(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        return split_tensor
    
    split_tensor = split_input(input_layer)
    
    def group_1x1_conv(input_tensor):
        return Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    def group_3x3_conv(input_tensor):
        return Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    def group_5x5_conv(input_tensor):
        return Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    group1 = group_1x1_conv(split_tensor[0])
    group2 = group_3x3_conv(split_tensor[1])
    group3 = group_5x5_conv(split_tensor[2])
    
    # Concatenate the outputs
    output_tensor = Concatenate()([group1, group2, group3])
    
    # Apply depthwise separable convolution
    dw = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    
    batch_norm = BatchNormalization()(dw)
    
    # Block 2
    def branch1(input_tensor):
        return Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    def branch2(input_tensor):
        return Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    def branch3(input_tensor):
        return Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    def branch4(input_tensor):
        return Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    def branch5(input_tensor):
        return MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_tensor)
    
    def branch6(input_tensor):
        return Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    def branch7(input_tensor):
        return Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    
    # Create the branches
    branch1_output = branch1(batch_norm)
    branch2_output = branch2(batch_norm)
    branch3_output = branch3(batch_norm)
    branch4_output = branch4(batch_norm)
    branch5_output = branch5(batch_norm)
    branch6_output = branch6(batch_norm)
    branch7_output = branch7(batch_norm)
    
    # Concatenate the outputs
    output_tensor = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output, branch5_output, branch6_output, branch7_output])
    
    # Apply batch normalization
    batch_norm = BatchNormalization()(output_tensor)
    
    # Flatten the output
    flatten_layer = Flatten()(batch_norm)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model