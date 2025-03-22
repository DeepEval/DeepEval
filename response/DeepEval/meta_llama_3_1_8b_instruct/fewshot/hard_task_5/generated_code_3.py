import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda, Reshape, Permute
import tensorflow as tf

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Split input into three groups and process each group separately
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=int(input_tensor.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=int(input_tensor.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=int(input_tensor.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Block 2: Reshape feature to achieve channel shuffling
    def block_2(input_tensor):
        shape = input_tensor.shape
        reshaped = Reshape(target_shape=(shape[1], shape[2], 3, int(shape[-1] / 3)))(input_tensor)
        permuted = Permute([1, 2, 4, 3])(reshaped)
        output_tensor = Reshape(target_shape=shape)(permuted)
        return output_tensor

    # Block 3: Process feature using 3x3 depthwise separable convolution
    def block_3(input_tensor):
        conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    block3_output = block_3(input_tensor=block2_output)

    # Main path ends here, now connect to input for branch
    branch = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine outputs from main path and branch using addition operation
    adding_layer = Add()([block3_output, branch])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model