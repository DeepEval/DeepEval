import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Lambda, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main path: splitting the input
    def main_path(input_tensor):
        # Split input into three groups along the last dimension
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # First group remains unchanged
        group1 = input_groups[0]

        # Second group: 3x3 convolution
        group2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_groups[1])

        # Third group remains unchanged and merged with second group before passing through 3x3 convolution
        combined_group23 = Concatenate()([group2, input_groups[2]])
        group23_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(combined_group23)

        # Concatenate all three groups to form the output of the main path
        main_path_output = Concatenate()([group1, group2, group23_conv])
        return main_path_output

    # Branch path: 1x1 convolution
    def branch_path(input_tensor):
        branch_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_output

    # Obtain outputs from both paths
    main_output = main_path(input_tensor=input_layer)
    branch_output = branch_path(input_tensor=input_layer)

    # Fuse outputs from both paths using addition
    fused_output = Add()([main_output, branch_output])

    # Final classification layers
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model