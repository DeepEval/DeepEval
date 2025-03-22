import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Split and Separable Convolutions
    def split_and_convolve(input_tensor):
        # Splitting channels into three groups
        split1, split2, split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Separable Convolutions with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split1)
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split2)
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split3)

        # Concatenating the outputs
        return Concatenate()([conv1, conv2, conv3])

    block1_output = split_and_convolve(input_layer)

    # Second Block: Multiple Branches
    def multi_branch_block(input_tensor):
        # Branch 1: 3x3 Convolution
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)

        # Branch 2: 1x1 Convolution followed by two 3x3 Convolutions
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

        # Branch 3: Max Pooling
        branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

        # Concatenating all branches
        return Concatenate()([branch1, branch2, branch3])

    block2_output = multi_branch_block(block1_output)

    # Global Average Pooling followed by Fully Connected Layer
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model