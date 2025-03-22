import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, Flatten, Dense, DepthwiseConv2D, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups
        split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv_layers = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split) for split in split_layers]
        # Concatenate the outputs
        output_tensor = Concatenate()(conv_layers)
        return output_tensor

    def block_2(input_tensor):
        # Get the shape and perform reshaping and permutation
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        reshaped = Reshape(target_shape=(height, width, 3, channels // 3))(input_tensor)
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))(reshaped)
        output_tensor = Reshape(target_shape=(height, width, channels))(permuted)
        return output_tensor

    def block_3(input_tensor):
        # Apply depthwise separable convolution
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Main path
    block1_output_1 = block_1(input_layer)
    block2_output = block_2(block1_output_1)
    block3_output = block_3(block2_output)
    block1_output_2 = block_1(block3_output)

    # Branch path
    branch_path = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    branch_flattened = Flatten()(branch_path)

    # Concatenate main path and branch path
    main_flattened = Flatten()(block1_output_2)
    concatenated = Concatenate()([main_flattened, branch_flattened])

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(concatenated)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model