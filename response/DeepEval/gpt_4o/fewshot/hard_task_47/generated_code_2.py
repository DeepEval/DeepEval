import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Split into three groups and apply depthwise separable convolutions
    def block_1(input_tensor):
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
        bn1 = BatchNormalization()(conv1)
        
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
        bn2 = BatchNormalization()(conv2)
        
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_groups[2])
        bn3 = BatchNormalization()(conv3)
        
        output_tensor = Concatenate()([bn1, bn2, bn3])
        return output_tensor
    
    # Second block: Multiple branches with different convolutional operations
    def block_2(input_tensor):
        # Branch 1
        branch1_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1_conv1)
        
        # Branch 2
        branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2_conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2_conv1)
        branch2_conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2_conv2)
        branch2_conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_conv3)
        
        # Branch 3
        branch3_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate branches
        output_tensor = Concatenate()([branch1_conv2, branch2_conv4, branch3_pool])
        return output_tensor
    
    # Applying blocks
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    
    # Final classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model