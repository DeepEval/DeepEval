import keras
from keras.layers import Input, Conv2D, Add, Lambda, DepthwiseConv2D, Dense, Flatten, Concatenate
import tensorflow as tf

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB)
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Dual-path structure with main and branch paths
    def dual_path_block(input_tensor):
        # Main path
        main_path_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path_conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)
        main_path_conv3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_conv2)
        
        # Branch path - directly from input
        branch_path = input_tensor
        
        # Addition of main and branch paths
        output_tensor = Add()([main_path_conv3, branch_path])
        
        return output_tensor

    # Second Block: Split input into three groups and apply depthwise separable convolutions
    def separable_conv_block(input_tensor):
        # Split input into 3 groups along the channel dimension
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply depthwise separable convolutions with different kernel sizes
        path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_groups[0])
        path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_groups[1])
        path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_groups[2])
        
        # Concatenate the outputs
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor

    # Apply the first block
    block1_output = dual_path_block(input_layer)
    
    # Apply the second block
    block2_output = separable_conv_block(block1_output)

    # Flatten the output
    flatten_layer = Flatten()(block2_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model