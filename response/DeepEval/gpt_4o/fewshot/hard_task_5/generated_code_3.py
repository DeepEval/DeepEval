import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        group_convs = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(group) for group in groups]
        output_tensor = Concatenate(axis=-1)(group_convs)
        return output_tensor

    def block_2(input_tensor):
        h, w, c = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
        reshaped = tf.reshape(input_tensor, (-1, h, w, 3, c // 3))
        permuted = tf.transpose(reshaped, perm=[0, 1, 2, 4, 3])
        reshuffled = tf.reshape(permuted, (-1, h, w, c))
        return reshuffled

    def block_3(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    # Main path
    block1_output_1 = block_1(input_layer)
    block2_output = block_2(block1_output_1)
    block3_output = block_3(block2_output)
    block1_output_2 = block_1(block3_output)

    # Branch path (direct connection to input)
    branch_path = input_layer

    # Combine paths
    combined_output = Add()([block1_output_2, branch_path])

    # Classification layer
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model