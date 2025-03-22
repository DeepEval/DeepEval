import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    def main_path(input_tensor):
        # Split the input along the channel axis into 3 groups
        split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Apply depthwise separable convolutions with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_channels[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_channels[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_channels[2])

        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Branch path
    def branch_path(input_tensor):
        # 1x1 convolution to align the number of output channels with the main path
        output_tensor = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Get outputs from main and branch paths
    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)

    # Add the outputs from both paths
    added_output = Add()([main_path_output, branch_path_output])

    # Fully connected layers for classification
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model