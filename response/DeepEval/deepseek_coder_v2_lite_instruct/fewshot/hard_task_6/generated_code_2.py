import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Concatenate, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1 (repeated)
    def block_1(input_tensor):
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        shape = input_tensor.shape
        reshaped = Reshape(target_shape=(shape[1], shape[2], 3, shape[3] // 3))(input_tensor)
        permuted = Lambda(lambda x: tf.transpose(x, [0, 1, 2, 4, 3]))(reshaped)
        output_tensor = Reshape(target_shape=(shape[1], shape[2], shape[3] * 3))(permuted)
        return output_tensor

    # Block 3
    def block_3(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    # Apply Blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    block3_output = block_3(input_tensor=block2_output)
    block1_repeated_output = block_1(input_tensor=block3_output)

    # Branch path
    branch_path = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    branch_path = Flatten()(branch_path)

    # Merge main path and branch path
    merged_output = Concatenate()([block1_repeated_output, branch_path])

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(merged_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model