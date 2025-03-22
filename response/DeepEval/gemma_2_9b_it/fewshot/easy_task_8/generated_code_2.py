import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    depthwise_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(depthwise_conv)
    pointwise_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(pointwise_conv)
    flatten_layer = Flatten()(dropout2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model