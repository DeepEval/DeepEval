import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    block_output1 = _block(input_layer)
    block_output2 = _block(input_layer)
    block_output3 = _block(input_layer)
    concat_layer = Concatenate()([block_output1, block_output2, block_output3])
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(units=10, activation='softmax')(dropout1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model

def _block(input_tensor):

    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    dropout = Dropout(0.2)(conv)
    return dropout