import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    conv2 = DepthwiseSeparableConv2D(64, (3, 3), padding='valid', activation='relu')(conv1)
    conv3 = Conv2D(64, (1, 1), strides=1, padding='same', activation='relu')(conv2)
    output_layer = Flatten()(Add()([conv1, conv3, input_layer]))
    output_layer = Dense(128, activation='relu')(output_layer)
    output_layer = Dense(10, activation='softmax')(output_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model