import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Concatenate, DepthwiseConv2D, Reshape, Permute
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1x1_1 = Conv2D(filters=int(split_groups[0].shape[-1]/3), kernel_size=(1, 1), padding='same', activation='relu')(split_groups[0])
        conv1x1_2 = Conv2D(filters=int(split_groups[1].shape[-1]/3), kernel_size=(1, 1), padding='same', activation='relu')(split_groups[1])
        conv1x1_3 = Conv2D(filters=int(split_groups[2].shape[-1]/3), kernel_size=(1, 1), padding='same', activation='relu')(split_groups[2])
        fused_features = Concatenate(axis=-1)([conv1x1_1, conv1x1_2, conv1x1_3])
        return fused_features

    def block_2(input_tensor):
        height, width, channels = input_tensor.shape[1:4]
        reshaped = Reshape(target_shape=(height, width, 3, int(channels/3)))(input_tensor)
        permuted = Permute((1, 2, 4, 3))(reshaped)
        reshaped_back = Reshape(target_shape=(height, width, channels))(permuted)
        return reshaped_back

    def block_3(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    main_path = block_1(input_tensor=input_layer)
    main_path = block_1(input_tensor=main_path)
    main_path = block_2(input_tensor=main_path)
    main_path = block_3(input_tensor=main_path)

    branch = input_layer
    added = Add()([main_path, branch])

    flattened = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model