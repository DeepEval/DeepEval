import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Dense, Add, Lambda, Flatten
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block_1(input_tensor):
        # Split input into 3 groups
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Apply 1x1 convolution on each group to reduce channels
        conv_layers = [Conv2D(filters=x.get_shape().as_list()[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(x) for x in split_layer]
        # Concatenate the processed groups
        concatenated = tf.keras.layers.Concatenate(axis=-1)(conv_layers)
        return concatenated
    
    # Block 2
    def block_2(input_tensor):
        # Get input shape
        shape = tf.shape(input_tensor)
        # Reshape to (height, width, groups, channels_per_group)
        reshaped = tf.keras.layers.Reshape((shape[1], shape[2], 3, shape[3] // 3))(input_tensor)
        # Swap the third and fourth dimension
        permuted = tf.keras.layers.Permute((1, 2, 4, 3))(reshaped)
        # Reshape back to original shape
        reshuffled = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]))(permuted)
        return reshuffled
    
    # Block 3
    def block_3(input_tensor):
        # Apply depthwise separable convolution
        separable_conv = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return separable_conv

    # Main Path
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    block3_output = block_3(block2_output)
    block1_repeat_output = block_1(block3_output)
    
    # Branch directly from input
    branch_output = input_layer
    
    # Combine main path and branch
    combined_output = Add()([block1_repeat_output, branch_output])
    
    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model