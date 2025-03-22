import keras
from keras.layers import Input, Conv2D, Add, Lambda, Concatenate, DepthwiseConv2D, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Dual-path structure
    def dual_path_block(input_tensor):
        # Main path
        main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
        main_path = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(main_path)

        # Branch path
        branch_path = input_tensor

        # Combine paths
        output_tensor = Add()([main_path, branch_path])

        return output_tensor

    # Second Block: Channel-split structure
    def channel_split_block(input_tensor):
        # Split input into 3 groups along the channels
        group1, group2, group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with depthwise separable conv layers
        group1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(group1)
        group2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(group2)
        group3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(group3)

        # Concatenate the outputs
        output_tensor = Concatenate()([group1, group2, group3])

        return output_tensor

    # Connect layers
    block1_output = dual_path_block(input_layer)
    block2_output = channel_split_block(block1_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model