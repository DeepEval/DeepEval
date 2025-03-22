import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2D, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
    batch_norm = BatchNormalization()(conv2)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model