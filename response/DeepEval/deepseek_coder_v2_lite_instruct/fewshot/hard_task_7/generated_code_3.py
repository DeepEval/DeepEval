import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dense, Flatten, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer to adjust dimensions
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    def block_1(input_tensor):
        # Split the input into two groups
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group operations
        group1_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_groups[0])
        group1_depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1_conv1x1)
        group1_final_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1_depthwise)
        
        # Second group remains unchanged
        group2 = split_groups[1]
        
        # Concatenate the outputs of both groups
        concatenated = Concatenate()([group1_final_conv1x1, group2])
        
        return concatenated

    block1_output = block_1(conv_initial)

    # Block 2
    def block_2(input_tensor):
        # Get the shape of the input
        input_shape = tf.shape(input_tensor)
        
        # Reshape into four groups
        reshaped_groups = Reshape(target_shape=(input_shape[1], input_shape[2], 2, input_shape[3] // 2))(input_tensor)
        
        # Permute the dimensions to swap the third and fourth dimensions
        permuted = tf.transpose(reshaped_groups, perm=[0, 1, 2, 4, 3])
        
        # Reshape back to the original shape to achieve channel shuffling
        final_shape = (input_shape[1], input_shape[2], input_shape[3] * 2)
        reshaped = Reshape(target_shape=final_shape)(permuted)
        
        return reshaped

    block2_output = block_2(block1_output)

    # Flatten the final output
    flattened = Flatten()(block2_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model