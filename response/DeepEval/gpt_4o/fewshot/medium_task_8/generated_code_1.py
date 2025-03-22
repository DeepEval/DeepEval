import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        # Split the input into three groups along the last dimension
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # First group remains unchanged
        group1 = inputs_groups[0]

        # Second group undergoes feature extraction via a 3x3 convolutional layer
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])

        # Combine the second group with the third group
        combined_group = Add()([group2, inputs_groups[2]])

        # Additional 3x3 convolution on the combined output
        combined_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)

        # Concatenate all three groups
        output_tensor = Concatenate()([group1, group2, combined_conv])

        return output_tensor

    def branch_path(input_tensor):
        # Process the input with a 1x1 convolutional layer
        output_tensor = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Get outputs from both paths
    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)

    # Fuse the outputs using addition
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten and add a fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model