import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting and processing with depthwise separable convolutions
    def block_1(input_tensor):
        # Split the input into three groups along the last dimension
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply depthwise separable convolutions with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
        conv1 = BatchNormalization()(conv1)
        
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
        conv2 = BatchNormalization()(conv2)
        
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_groups[2])
        conv3 = BatchNormalization()(conv3)
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Block 2: Multiple branches for feature extraction
    def block_2(input_tensor):
        # Branch 1: 1x1 convolution followed by 3x3 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

        # Branch 2: 1x1 convolution followed by 1x7, 7x1 and 3x3 convolutions
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        # Branch 3: Average pooling
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

        # Concatenate all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    # Connect the blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Final classification layers
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model