import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Add, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
        dropout = Dropout(0.2)(conv)
        return conv

    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)

    adding_layer = Add()([branch1, branch2, branch3])

    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(units=10, activation='softmax')(dropout1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model