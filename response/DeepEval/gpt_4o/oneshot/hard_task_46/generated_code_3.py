import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Split and apply separable convolution
    def split_and_separable_conv(input_tensor):
        # Split the input tensor into 3 parts along the channel dimension
        split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_channels[0])
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_channels[1])
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_channels[2])
        
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
    
    block1_output = split_and_separable_conv(input_layer)
    
    # Second block: Multiple branches for feature extraction
    def feature_extraction_block(input_tensor):
        # Branch 1: 3x3 Convolution
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 Conv followed by two 3x3 Convolutions
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: Max Pooling
        branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor
    
    block2_output = feature_extraction_block(block1_output)
    
    # Global Average Pooling and Fully Connected Layer
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model