import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, Dense, Reshape, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into 3 groups
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Process each group with a 1x1 convolutional layer
        processed_groups = [Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(group) for group in split_groups]
        # Concatenate the processed groups
        output_tensor = Concatenate(axis=-1)(processed_groups)
        return output_tensor

    def block_2(input_tensor):
        # Get the shape of the input
        height, width, channels = input_tensor.shape[1:4]
        # Reshape to (height, width, groups, channels_per_group)
        reshaped = Reshape(target_shape=(height, width, 3, int(channels / 3)))(input_tensor)
        # Permute the dimensions to swap groups and channels
        permuted = tf.transpose(reshaped, perm=[0, 1, 3, 2])
        # Reshape back to the original shape
        output_tensor = Reshape(target_shape=(height, width, channels))(permuted)
        return output_tensor

    def block_3(input_tensor):
        # Apply 3x3 depthwise separable convolution
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(input_tensor)
        return output_tensor

    # Main path
    main_path = input_layer
    for _ in range(3):  # Repeat Block 1, Block 2, and Block 3
        main_path = block_1(main_path)
        main_path = block_2(main_path)
        main_path = block_3(main_path)

    # Branch path
    branch_path = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    branch_path = Reshape(target_shape=(1, 1, 3 * 64))(branch_path)  # Adjust the shape to match the main path

    # Concatenate main path and branch path
    combined_output = Concatenate(axis=-1)([main_path, branch_path])

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(combined_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model