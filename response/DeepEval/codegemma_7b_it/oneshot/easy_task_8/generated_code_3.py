import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    depthwise_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', depth_multiplier=2, activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(depthwise_conv)
    dropout1 = Dropout(0.25)(max_pooling)

    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv)

    flatten_layer = Flatten()(dropout2)
    dense = Dense(units=128, activation='relu')(flatten_layer)
    dropout3 = Dropout(0.5)(dense)

    output_layer = Dense(units=10, activation='softmax')(dropout3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model