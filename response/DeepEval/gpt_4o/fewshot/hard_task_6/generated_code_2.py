import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, DepthwiseConv2D, Dense, Reshape, Activation, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def block_1(input_tensor):
        # Splitting the input into three groups for processing
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # 1x1 convolutions on each group
        conv1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_groups[0])
        conv2 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_groups[1])
        conv3 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_groups[2])
        # Concatenating results from all paths
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # Obtain shape to determine dimensions
        height, width, channels = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
        # Reshaping for channel shuffling
        reshaped = Reshape((height, width, 3, channels // 3))(input_tensor)
        # Permuting dimensions to shuffle channels
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))(reshaped)
        # Reshaping back to original shape
        output_tensor = Reshape((height, width, channels))(permuted)
        return output_tensor

    def block_3(input_tensor):
        # Applying depthwise separable convolution
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Constructing the main path
    block1_1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_1_output)
    block3_output = block_3(input_tensor=block2_output)
    block1_2_output = block_1(input_tensor=block3_output)

    # Constructing the branch path
    branch_path = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    branch_path_flatten = Flatten()(branch_path)

    # Concatenating main path and branch path
    concatenated = Concatenate()([Flatten()(block1_2_output), branch_path_flatten])

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(concatenated)

    # Building the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model