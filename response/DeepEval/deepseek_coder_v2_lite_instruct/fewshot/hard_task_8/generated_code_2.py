import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Dense, Reshape, Permute
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # Primary path
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv3x3)
        
        # Branch path
        depthwise_conv3x3_branch = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv3x3_branch)
        
        # Concatenate along the channel dimension
        concatenated = Concatenate(axis=-1)([conv1x1_2, conv1x1_branch])
        return concatenated

    def block_2(input_tensor):
        # Get the shape of the input
        input_shape = tf.keras.backend.int_shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        
        # Reshape into groups
        reshaped = Reshape(target_shape=(height, width, 4, int(channels / 4)))(input_tensor)
        
        # Permute dimensions to achieve channel shuffling
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Flatten the reshaped tensor
        flattened = Flatten()(permuted)
        
        return flattened

    # Apply Block 1
    block1_output = block_1(input_layer)
    
    # Apply Block 2
    block2_output = block_2(block1_output)
    
    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(block2_output)

    # Construct and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model