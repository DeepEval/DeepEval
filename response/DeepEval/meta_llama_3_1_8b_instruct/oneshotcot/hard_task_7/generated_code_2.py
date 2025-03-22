import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Lambda, Reshape, Permute, Flatten, Dense
from tensorflow import split as tf_split
from tensorflow import shape as tf_shape

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    def block1(input_tensor):
        
        # Split the input into two groups along the last dimension
        split_tensor = Lambda(lambda x: tf_split(x, 2))(input_tensor)
        group1 = split_tensor[0]
        group2 = split_tensor[1]

        # Operations for the first group
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
        merged_group1 = Concatenate()([conv1, conv2])

        # The second group is passed through without modification
        merged_group2 = group2

        # Merge the outputs from both groups
        output_tensor = Concatenate()([merged_group1, merged_group2])

        return output_tensor

    block1_output = block1(conv)

    def block2(input_tensor):

        # Obtain the shape of the input tensor
        shape = Lambda(lambda x: tf_shape(x))(input_tensor)

        # Reshape the input into four groups, with a target shape of (height, width, groups, channels_per_group)
        reshape_tensor = Reshape((-1, 7, 7, 32 // 2, 2))(input_tensor)

        # Swap the third and fourth dimensions using permutation operations
        permute_tensor = Permute((1, 2, 4, 3, 5))(reshape_tensor)

        # Reshape the input back to its original shape to achieve channel shuffling
        reshaped_tensor = Reshape((7, 7, 32))(permute_tensor)

        # Flatten the output
        flatten_tensor = Flatten()(reshaped_tensor)

        # Pass the output through a fully connected layer for classification
        output_tensor = Dense(units=10, activation='softmax')(flatten_tensor)

        return output_tensor

    block2_output = block2(block1_output)

    model = keras.Model(inputs=input_layer, outputs=block2_output)

    return model