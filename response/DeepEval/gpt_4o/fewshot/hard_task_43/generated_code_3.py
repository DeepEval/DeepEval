import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)
        
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    def block_2(input_tensor):
        # Branch 1: <1x1 convolution, 3x3 convolution>
        branch1_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1_conv1)
        
        # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution>
        branch2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2_conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2_conv1)
        branch2_conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2_conv2)
        branch2_conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_conv3)
        
        # Branch 3: Average pooling
        branch3_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate all branches
        output_tensor = Concatenate()([branch1_conv2, branch2_conv4, branch3_pool])
        return output_tensor
    
    # Process through Block 1
    block1_output = block_1(input_tensor=input_layer)
    
    # Fully connected and reshape between Block 1 and Block 2
    dense = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(8, 8, 2))(dense)  # Adjust the reshape dimensions based on the output size
    
    # Process through Block 2
    block2_output = block_2(input_tensor=reshaped)
    
    # Final fully connected layers for classification
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model