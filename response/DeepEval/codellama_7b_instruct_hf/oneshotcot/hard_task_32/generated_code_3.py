import keras
from keras.layers import Input, Conv2D, MaxPooling2D, DepthwiseSeparableConv2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense


def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    def block(input_tensor):
        depthwise_separable = DepthwiseSeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout = Dropout(rate=0.2)(depthwise_separable)
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout)
        dropout = Dropout(rate=0.2)(conv)
        return dropout
    branch1 = block(branch1)
    branch2 = block(branch2)
    branch3 = block(branch3)
    concat = Concatenate()([branch1, branch2, branch3])
    bath_norm = BatchNormalization()(concat)
    flatten = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model