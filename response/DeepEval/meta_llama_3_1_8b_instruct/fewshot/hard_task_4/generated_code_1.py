import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Reshape, Concatenate, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    initial_conv = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)

    channel_attention = GlobalAveragePooling2D()(depthwise_conv)
    channel_attention = Dense(units=96, activation='relu')(channel_attention)
    channel_attention = Dense(units=96, activation='sigmoid')(channel_attention)
    channel_attention = Reshape(target_shape=(1, 1, 96))(channel_attention)
    channel_attention = Multiply()([channel_attention, depthwise_conv])

    final_conv = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_attention)
    combined = Concatenate()([initial_conv, final_conv])

    flatten = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model