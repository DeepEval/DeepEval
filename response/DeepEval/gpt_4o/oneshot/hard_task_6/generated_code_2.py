import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Dense, Lambda, SeparableConv2D, Flatten
from keras import backend as K

def dl_model():
    def block1(input_tensor):
        # Split the input into three groups
        def split(x):
            return tf.split(x, num_or_size_splits=3, axis=-1)

        split_layer = Lambda(split)(input_tensor)

        # Process each group with 1x1 convolutions
        path1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_layer[0])
        path2 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_layer[1])
        path3 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_layer[2])

        # Concatenate the processed groups
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    def block2(input_tensor):
        # Obtain the shape of the input tensor
        input_shape = K.int_shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        groups = 3

        # Reshape and shuffle channels
        def reshape_shuffle(x):
            x = K.reshape(x, (-1, height, width, groups, channels // groups))
            x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
            x = K.reshape(x, (-1, height, width, channels))
            return x

        output_tensor = Lambda(reshape_shuffle)(input_tensor)

        return output_tensor

    def block3(input_tensor):
        # Apply a 3x3 depthwise separable convolution
        output_tensor = SeparableConv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)

        return output_tensor

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    block1_output_1 = block1(input_layer)
    block2_output = block2(block1_output_1)
    block3_output = block3(block2_output)
    block1_output_2 = block1(block3_output)

    # Branch path
    branch_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)

    # Concatenate main path and branch path outputs
    combined_output = Concatenate()([block1_output_2, branch_output])

    # Final classification layer
    flatten = Flatten()(combined_output)
    dense = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model