from keras.models import Model
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Add, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    channels = input_layer.shape[-1]
    dw_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same')(input_layer)
    norm = LayerNormalization()(dw_conv)
    pw_conv1 = Dense(units=channels*3, activation='relu')(norm)
    pw_conv2 = Dense(units=channels)(pw_conv1)
    main_path = pw_conv2

    added = Add()([main_path, input_layer])
    dense_1 = Dense(units=128, activation='relu')(added)
    output_layer = Dense(units=10, activation='softmax')(dense_1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


