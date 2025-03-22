import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1: Splitting into groups and applying 1x1 convolution
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Block 2: Reshaping and performing channel shuffling
    def block_2(input_tensor):
        shape = tf.shape(input_tensor)
        reshaped = Reshape((shape[1], shape[2], 3, shape[-1] // 3))(input_tensor)  # (height, width, groups, channels_per_group)
        permuted = tf.transpose(reshaped, perm=[0, 1, 3, 2])  # Swap groups and channels
        output_tensor = Reshape((shape[1], shape[2], shape[-1]))(permuted)  # Reshape back to (height, width, channels)
        return output_tensor

    # Block 3: Applying depthwise separable convolution
    def block_3(input_tensor):
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Main Path
    block1_output_1 = block_1(input_layer)
    block2_output = block_2(block1_output_1)
    block3_output = block_3(block2_output)
    block1_output_2 = block_1(block3_output)

    # Branch Path
    branch_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)

    # Concatenate both paths
    concatenated_output = Concatenate()([block1_output_2, branch_output])

    # Fully connected layer for classification
    flatten_output = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model