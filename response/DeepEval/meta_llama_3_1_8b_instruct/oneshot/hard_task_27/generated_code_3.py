import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, LayerNormalization, Add, Dense
from keras.layers import GlobalAveragePooling2D, Reshape, Lambda, Concatenate

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same')(input_layer)
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv)
    layer_norm = LayerNormalization()(conv)

    flatten_layer = GlobalAveragePooling2D()(layer_norm)
    reshape_layer = Reshape((64,))(flatten_layer)

    def channel_transformation(input_tensor):
        dense1 = Dense(units=64, activation='relu')(input_tensor)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_tensor = Add()([dense1, dense2])
        return output_tensor

    transformed_features = channel_transformation(reshape_layer)
    original_input = Lambda(lambda x: x[:, :64])(reshape_layer)

    combined_features = Concatenate()([original_input, transformed_features])
    dense1 = Dense(units=64, activation='relu')(combined_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model