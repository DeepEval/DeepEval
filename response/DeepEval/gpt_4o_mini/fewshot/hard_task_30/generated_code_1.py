import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate, Lambda, DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)

    # First Block with Dual-Path Structure
    def block_1(input_tensor):
        # Main path
        main_path_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)
        main_path_output = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same')(main_path_conv2)  # Restore number of channels

        # Branch path
        branch_path_output = input_tensor  # Direct connection to the input

        # Combine both paths
        combined_output = Add()([main_path_output, branch_path_output])
        return combined_output

    # Second Block with Depthwise Separable Convolutions
    def block_2(input_tensor):
        # Split into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Depthwise separable convolutions
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])

        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Process through the blocks
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)

    # Flatten and Fully Connected Layers
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model