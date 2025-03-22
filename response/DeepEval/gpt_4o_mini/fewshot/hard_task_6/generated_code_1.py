import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, Dense, Reshape, DepthwiseConv2D
from keras.models import Model

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    def block_1(input_tensor):
        # Split the input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # Obtain shape and reshape
        shape = tf.shape(input_tensor)
        reshaped = Reshape(target_shape=(shape[1], shape[2], 3, -1))(input_tensor)  # groups=3
        permuted = tf.transpose(reshaped, perm=[0, 1, 3, 2])  # Swap groups and channels
        output_tensor = Reshape(target_shape=(shape[1], shape[2], shape[3]))(permuted)  # Reshape back
        return output_tensor

    def block_3(input_tensor):
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Main path
    block1_output_1 = block_1(input_layer)
    block2_output_1 = block_2(block1_output_1)
    block3_output_1 = block_3(block2_output_1)

    block1_output_2 = block_1(block3_output_1)
    block2_output_2 = block_2(block1_output_2)
    block3_output_2 = block_3(block2_output_2)

    # Branch path
    branch_output = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)

    # Concatenate outputs from the main path and branch path
    combined_output = Concatenate()([block3_output_2, branch_output])

    # Fully connected layer for classification
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model