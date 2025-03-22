import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block: Depthwise separable convolutions
    def depthwise_separable_conv(input_tensor, filters, kernel_size):
        # Depthwise convolution
        depthwise_conv = Conv2D(filters, kernel_size, padding='same', depthwise_constraint=True, activation='relu')(input_tensor)
        # Pointwise convolution
        pointwise_conv = Conv2D(filters, (1, 1), activation='relu')(depthwise_conv)
        return pointwise_conv

    # Apply depthwise separable convolutions to three groups
    conv1 = depthwise_separable_conv(input_layer, 32, (3, 3))
    conv2 = depthwise_separable_conv(input_layer, 32, (5, 5))
    conv3 = depthwise_separable_conv(input_layer, 32, (1, 1))

    # Concatenate outputs from the three groups
    concat_layer = Concatenate()([conv1, conv2, conv3])

    # Batch normalization
    batch_norm = BatchNormalization()(concat_layer)

    # Second block: Multiple branches for feature extraction
    def branch(input_tensor, filters, kernel_sizes):
        # 1x1 convolution
        conv_1x1 = Conv2D(filters, (1, 1), activation='relu')(input_tensor)
        # 3x3 convolution
        conv_3x3 = Conv2D(filters, kernel_sizes[0], padding='same', activation='relu')(conv_1x1)
        # 1x7 and 7x1 convolution
        conv_1x7 = Conv2D(filters, (1, 7), padding='same', activation='relu')(conv_1x1)
        conv_7x1 = Conv2D(filters, (7, 1), padding='same', activation='relu')(conv_1x7)
        # Average pooling
        avg_pool = tf.reduce_mean(input_tensor, [1, 2], keepdims=True)
        avg_pool = Conv2D(filters, kernel_sizes[1], padding='same', activation='relu')(avg_pool)
        return Concatenate()([conv_3x3, conv_7x1, avg_pool])

    # Apply the branch function to each set of operations
    branch1 = branch(batch_norm, 64, [(3, 3), (5, 5)])
    branch2 = branch(batch_norm, 64, [(1, 1), (3, 3)])
    branch3 = branch(batch_norm, 64, [(1, 1), (1, 7), (7, 1), (3, 3)])

    # Concatenate outputs from all branches
    concat_branch_output = Concatenate()([branch1, branch2, branch3])

    # Flatten the output
    flatten_layer = Flatten()(concat_branch_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model