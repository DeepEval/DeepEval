import keras
from keras.layers import Input, Conv2D, SeparableConv2D, AveragePooling2D, MaxPooling2D, Dense, BatchNormalization, Flatten, Concatenate, Lambda
import tensorflow as tf

def dl_model():

    # Input layer for CIFAR-10 (32x32 RGB images)
    input_layer = Input(shape=(32, 32, 3))

    # First block: Split input into three groups and apply separable convolutions
    def first_block(input_tensor):
        # Split the input along the channel dimension into 3 groups
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Apply depthwise separable convolutions with different kernel sizes
        conv_1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv_3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv_5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])

        # Concatenate the outputs
        concat_layer = Concatenate()([conv_1x1, conv_3x3, conv_5x5])

        # Apply batch normalization
        block_output = BatchNormalization()(concat_layer)

        return block_output

    block1_output = first_block(input_layer)

    # Second block: Multiple branches for feature extraction
    def second_block(input_tensor):
        # Branch 1: 1x1 convolution followed by 3x3 convolution
        branch1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1_3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch1_1x1)

        # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, followed by 3x3 convolution
        branch2_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2_1x7 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(branch2_1x1)
        branch2_7x1 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(branch2_1x7)
        branch2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2_7x1)

        # Branch 3: Average pooling
        branch3_avg_pool = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)

        # Concatenate the outputs from the branches
        concat_layer = Concatenate()([branch1_3x3, branch2_3x3, branch3_avg_pool])

        return concat_layer

    block2_output = second_block(block1_output)

    # Flatten the output
    flatten_layer = Flatten()(block2_output)

    # Dense layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model