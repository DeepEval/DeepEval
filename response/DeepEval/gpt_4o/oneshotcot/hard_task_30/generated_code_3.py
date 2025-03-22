import keras
from keras.layers import Input, Conv2D, Add, Lambda, Concatenate, Dense, Flatten, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Dual-path structure
    def first_block(input_tensor):
        # Main path
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
        
        # Branch path (identity)
        branch = input_tensor

        # Add main path and branch path
        block_output = Add()([conv3, branch])
        return block_output

    # Second Block: Split and depthwise separable convolutions
    def second_block(input_tensor):
        # Split the input along the channel dimension
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Process each split with depthwise separable convolutions
        path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(splits[2])

        # Concatenate the results from the three paths
        block_output = Concatenate()([path1, path2, path3])
        return block_output

    # Apply the blocks to the input layer
    block1_output = first_block(input_layer)
    block2_output = second_block(block1_output)
    
    # Final dense layers for classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model