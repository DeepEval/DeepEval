import keras
import tensorflow as tf
from keras.layers import Input, Lambda, SeparableConv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        # Split the input into three groups along the channel dimension
        inputs_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Apply separable convolutional layers with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])

        # Concatenate the outputs to form the main path output
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Branch path
    def branch_path(input_tensor):
        # Apply a 1x1 convolution to align the number of output channels
        output_tensor = SeparableConv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Calculate the outputs of the main and branch paths
    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)

    # Fuse outputs from both paths
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten the output and pass through two fully connected layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model