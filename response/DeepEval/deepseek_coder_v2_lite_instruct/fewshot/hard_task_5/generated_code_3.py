import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Concatenate, Reshape, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        # Split the input into three groups
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Process each group with a 1x1 convolutional layer
        processed_groups = [Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1, 1), padding='same', activation='relu')(group) for group in split_layer]
        # Concatenate the processed groups along the channel dimension
        fused_features = Concatenate(axis=-1)(processed_groups)
        return fused_features

    def block_2(input_tensor):
        # Get the shape of the input tensor
        height, width, channels = input_tensor.shape[1:4]
        # Reshape the tensor into three groups
        reshaped_groups = Reshape(target_shape=(height, width, 3, channels//3))(input_tensor)
        # Permute the dimensions to swap the third and fourth dimensions
        permuted_groups = tf.transpose(reshaped_groups, [0, 1, 3, 2])
        # Reshape back to the original shape
        shuffled_features = Reshape(target_shape=(height, width, channels))(permuted_groups)
        return shuffled_features

    def block_3(input_tensor):
        # Process the input with a 3x3 depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    # Apply Block 1 to the input
    block1_output = block_1(input_tensor=input_layer)
    # Apply Block 2 to the output of Block 1
    block2_output = block_2(input_tensor=block1_output)
    # Apply Block 3 to the output of Block 2
    block3_output = block_3(input_tensor=block2_output)
    # Apply Block 1 again to the output of Block 3
    block1_again_output = block_1(input_tensor=block3_output)

    # Branch from the input to the model
    branch = input_layer

    # Add the main path output and the branch output
    added_output = Add()([block1_again_output, branch])

    # Flatten the output and pass it through a fully connected layer
    flattened_output = Flatten()(added_output)
    final_output = Dense(units=10, activation='softmax')(flattened_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model