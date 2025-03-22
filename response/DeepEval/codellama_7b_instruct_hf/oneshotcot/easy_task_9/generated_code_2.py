from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    depthwise_conv = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    flatten_layer = Flatten()(max_pooling)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model