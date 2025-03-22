import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # Main path
    def main_path(input_tensor):
        # Splitting the input into 3 groups along the channel
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Applying depthwise separable convolutions with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])

        # Concatenating the outputs from the three convolutions
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Branch path
    def branch_path(input_tensor):
        # Using a 1x1 convolution to align the number of output channels
        conv_branch = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv_branch

    # Getting outputs from both paths
    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)

    # Adding the outputs from both paths
    added_output = Add()([main_path_output, branch_path_output])

    # Fully connected layers for classification
    flatten = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model