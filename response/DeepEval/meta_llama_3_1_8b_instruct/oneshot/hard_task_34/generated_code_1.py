import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def feature_extract_block(input_tensor):

        conv = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv)
        output_tensor = Concatenate()([input_tensor, conv])

        return output_tensor

    feature_extract_block_output = feature_extract_block(input_layer)
    for _ in range(3):
        feature_extract_block_output = feature_extract_block(feature_extract_block_output)

    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(feature_extract_block_output)

    fusion = Add()([feature_extract_block_output, conv_branch])
    bath_norm = BatchNormalization()(fusion)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model