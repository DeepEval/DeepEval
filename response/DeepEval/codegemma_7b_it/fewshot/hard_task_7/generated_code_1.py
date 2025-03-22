import keras
from keras.layers import Input, Conv2D, Lambda, Dense, Concatenate, Reshape, Permute

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv_dw = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(conv1)
        conv_pw = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_dw)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        output_tensor = Concatenate()([conv1, conv2])
        return output_tensor

    block1_output = block_1(input_tensor=conv_initial)

    # Block 2
    def block_2(input_tensor):
        input_shape = keras.backend.int_shape(input_tensor)
        input_shape_reshaped = input_shape[1:3] + (input_shape[3] // 4, 4)
        reshaped = Reshape(target_shape=input_shape_reshaped)(input_tensor)
        reshaped_swapped = Permute((2, 3, 1, 0))(reshaped)
        reshaped_swapped_reshaped = Reshape(target_shape=input_shape)(reshaped_swapped)
        output_tensor = reshaped_swapped_reshaped
        return output_tensor

    block2_output = block_2(input_tensor=block1_output)

    # Final layer
    flatten = keras.layers.Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model