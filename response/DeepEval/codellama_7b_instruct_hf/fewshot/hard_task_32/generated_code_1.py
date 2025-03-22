import keras
from keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization, Flatten, Conv2D, DepthwiseSeparableConv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        depthwise = DepthwiseSeparableConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        dropout = Dropout(rate=0.2)(conv2)
        return dropout

    branch1 = block(input_tensor=input_layer)
    branch2 = block(input_tensor=input_layer)
    branch3 = block(input_tensor=input_layer)
    concatenated = Concatenate()([branch1, branch2, branch3])
    flattened = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model