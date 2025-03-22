import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Lambda, Flatten, Dense, Concatenate, DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        # Main path
        main_path_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)
        main_path_conv3 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv2)

        # Branch path
        branch_path = input_tensor

        # Combine paths
        output_tensor = Add()([main_path_conv3, branch_path])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        # Split the input tensor into 3 groups along the channel dimension
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply depthwise separable convolutions
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])

        # Concatenate outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Processing through the two blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Fully connected layers for classification
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model