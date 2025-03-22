from keras.models import Model
from keras.layers import Input, LayerNormalization, DepthwiseConv2D, Conv2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    channels = input_layer.shape[-1]
    dw_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1),padding='same')(input_layer)
    norm = LayerNormalization()(dw_conv)

    pw_conv = Conv2D(filters=4*channels, kernel_size=(1, 1), strides=(1, 1),padding='same',activation='gelu')(norm)
    main_path = Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1),padding='same')(pw_conv)

    added = Add()([main_path, input_layer])
    
    flatten_output = Flatten()(added)
    dense = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
