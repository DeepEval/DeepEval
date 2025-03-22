import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, DepthwiseConv2D, Dense, Reshape, Permute

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=int(input_tensor.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=int(input_tensor.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=int(input_tensor.shape[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        _, height, width, _ = input_tensor.shape.as_list()
        reshaped = Reshape(target_shape=(height, width, 3, int(input_tensor.shape[-1] / 3)))(input_tensor)
        permuted = Permute((1, 2, 4, 3))(reshaped)
        reshaped = Reshape(target_shape=(height, width, int(input_tensor.shape[-1] / 3)))(permuted)
        return reshaped

    def block_3(input_tensor):
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    main_path = block_1(input_layer)
    main_path = block_2(main_path)
    main_path = block_3(main_path)
    main_path = block_1(main_path)
    main_path = block_2(main_path)
    main_path = block_3(main_path)

    branch_path = AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='valid')(input_layer)
    combined_path = Concatenate()([main_path, branch_path])

    flatten_layer = Flatten()(combined_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model