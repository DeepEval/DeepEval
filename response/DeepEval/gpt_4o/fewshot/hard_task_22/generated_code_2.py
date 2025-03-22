import keras
import tensorflow as tf
from keras.layers import Input, Lambda, SeparableConv2D, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        # Split input into 3 groups along the channel dimension
        input_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Apply separable convolutions of different kernel sizes to each group
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_groups[2])

        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def branch_path(input_tensor):
        # Apply 1x1 convolution to align the output channels with the main path
        output_tensor = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Get outputs from both paths
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)

    # Fuse the outputs from both paths
    fused_output = Add()([main_output, branch_output])

    # Flatten the fused output
    flatten_layer = Flatten()(fused_output)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model