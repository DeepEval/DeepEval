import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, DepthwiseConv2D, Flatten, Dense, Reshape

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split input into 3 groups
        inputs_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Apply 1x1 convolution to each group
        conv1 = Conv2D(filters=int(input_tensor.shape[-1] // 3), kernel_size=(1, 1), activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=int(input_tensor.shape[-1] // 3), kernel_size=(1, 1), activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=int(input_tensor.shape[-1] // 3), kernel_size=(1, 1), activation='relu')(inputs_groups[2])
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # Get shape of features
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        # Reshape for channel shuffling
        reshaped = Reshape((height, width, 3, channels // 3))(input_tensor)
        # Permute the dimensions
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))(reshaped)
        # Reshape back to original shape
        output_tensor = Reshape((height, width, channels))(permuted)
        return output_tensor

    def block_3(input_tensor):
        # Apply 3x3 depthwise separable convolution
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Main path
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    block3_output = block_3(input_tensor=block2_output)
    block1_repeat_output = block_1(input_tensor=block3_output)

    # Branch path
    branch_output = AveragePooling2D(pool_size=(2, 2))(input_layer)
    
    # Concatenate main path and branch path
    concatenated = Concatenate()([block1_repeat_output, branch_output])
    
    # Final classification layer
    flatten = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model