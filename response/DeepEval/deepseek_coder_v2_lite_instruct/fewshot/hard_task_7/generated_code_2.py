import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Flatten, Dense, DepthwiseConv2D, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer in Block 1
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    def block_1(input_tensor):
        # Split the input into two groups
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group operations
        group1_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_groups[0])
        group1_depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1_conv1x1)
        group1_final_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1_depthwise)
        
        # Second group remains unchanged
        group2 = split_groups[1]
        
        # Merge the outputs
        merged = Concatenate()([group1_final_conv1x1, group2])
        return merged

    block1_output = block_1(conv_initial)

    # Block 2
    def block_2(input_tensor):
        # Get the shape of the input
        input_shape = tf.keras.backend.int_shape(input_tensor)
        
        # Reshape into four groups
        reshaped_groups = Reshape(target_shape=(input_shape[1], input_shape[2], 2, int(input_shape[3]/2)))(input_tensor)
        
        # Permute dimensions to achieve channel shuffling
        permuted = Lambda(lambda x: tf.transpose(x, [0, 1, 2, 4, 3]))(reshaped_groups)
        
        # Flatten the result
        flattened = Flatten()(permuted)
        
        return flattened

    block2_output = block_2(block1_output)

    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(block2_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model