import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Permute, Reshape, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Primary path
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_separable_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(primary_path)
    depthwise_separable_conv2 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_separable_conv1)

    # Block 1: Branch path
    branch_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_separable_conv3 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

    # Block 1: Merge primary and branch paths
    merged_path = Concatenate(axis=3)([depthwise_separable_conv2, depthwise_separable_conv3])

    # Block 2: Reshape and shuffle channels
    reshaped_path = Reshape(target_shape=(28, 28, 4))(merged_path)
    permuted_path = Permute(dims=(0, 2, 3, 1))(reshaped_path)
    shuffled_path = Reshape(target_shape=(28, 28, 4))(permuted_path)

    # Fully connected layer
    flattened_path = Flatten()(shuffled_path)
    output_layer = Dense(units=10, activation='softmax')(flattened_path)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model