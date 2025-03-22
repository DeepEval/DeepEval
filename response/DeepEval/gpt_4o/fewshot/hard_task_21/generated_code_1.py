import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Add, Dense, Flatten, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        # Split the input into three groups along the channel dimension
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Apply depthwise separable convolutions with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])

        # Concatenate the outputs of the depthwise convolutions
        output_tensor = Concatenate()([conv1, conv3, conv5])
        return output_tensor

    def branch_path(input_tensor):
        # Align the number of output channels with those of the main path
        output_tensor = Conv2D(filters=3*32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Create the main and branch paths
    main_output = main_path(input_tensor=input_layer)
    branch_output = branch_path(input_tensor=input_layer)

    # Combine the outputs from the main and branch paths
    combined_output = Add()([main_output, branch_output])

    # Flatten and add fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model