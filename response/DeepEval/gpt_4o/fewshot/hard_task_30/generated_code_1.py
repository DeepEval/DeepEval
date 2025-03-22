import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Lambda, Concatenate, DepthwiseConv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Dual-path structure
    def block_1(input_tensor):
        # Main path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        main_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

        # Branch path
        branch_path = input_tensor

        # Combine paths
        output_tensor = Add()([main_path, branch_path])
        return output_tensor

    # Second Block: Depthwise separable convolutions
    def block_2(input_tensor):
        # Split input into three groups along the channel dimension
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply depthwise separable convolutions with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Build the model using the defined blocks
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)

    # Flatten and add fully connected layers for classification
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model