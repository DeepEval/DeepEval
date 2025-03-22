import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Reshape, Permute, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Split the input into three groups
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        path1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        path2 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(splits[1])
        path3 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(splits[2])
        output_tensor = Concatenate(axis=-1)([path1, path2, path3])
        return output_tensor
    
    block1_output = block1(input_layer)

    # Block 2
    def block2(input_tensor):
        # Reshape into (height, width, groups, channels_per_group)
        batch_size = tf.shape(input_tensor)[0]
        height = input_tensor.shape[1]
        width = input_tensor.shape[2]
        channels = input_tensor.shape[3]
        
        groups = 3
        channels_per_group = channels // groups
        
        reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)
        permuted = Permute((0, 1, 3, 2))(reshaped)  # Swap the last two dimensions
        output_tensor = Reshape((height, width, channels))(permuted)  # Reshape back to original
        return output_tensor
    
    block2_output = block2(block1_output)

    # Block 3
    def block3(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv
    
    block3_output = block3(block2_output)

    # Branch connecting directly to input
    branch_output = input_layer

    # Combine the main path and branch output
    combined_output = Add()([block3_output, branch_output])

    # Final flatten and dense layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model