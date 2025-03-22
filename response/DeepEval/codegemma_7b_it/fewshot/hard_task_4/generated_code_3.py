import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, GlobalAveragePooling2D, Dense, Reshape, concatenate

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def initial_features(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        return dw_conv

    def channel_attention(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=16, activation='relu')(avg_pool)
        dense2 = Dense(units=32, activation='relu')(dense1)
        dense3 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense2)
        reshaped_dense = Reshape(target_shape=input_tensor.shape[1:-1] + (input_tensor.shape[-1],))(dense3)
        return reshaped_dense

    def channel_attention_weighted(input_tensor):
        ca_weights = channel_attention(input_tensor)
        return multiply([input_tensor, ca_weights])

    def reduce_dimensionality(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    initial_features_output = initial_features(input_tensor=input_layer)
    ca_weighted = channel_attention_weighted(initial_features_output)
    reduced_dimensionality_output = reduce_dimensionality(ca_weighted)
    concat = concatenate([initial_features_output, reduced_dimensionality_output])
    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model