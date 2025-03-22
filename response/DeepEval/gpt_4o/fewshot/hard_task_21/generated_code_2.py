import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Concatenate, Add, DepthwiseConv2D, Conv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        # Split the input into three groups along the channel axis
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Depthwise separable convolutions with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(input_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(input_groups[2])
        
        # Concatenate the outputs to form the main path output
        main_output = Concatenate()([conv1, conv2, conv3])
        return main_output

    # Branch path
    def branch_path(input_tensor):
        # 1x1 convolution to align the number of output channels
        branch_output = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_output

    # Get outputs from both paths
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)

    # Add outputs from the main path and branch path
    added_output = Add()([main_output, branch_output])

    # Fully connected layers for classification
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model