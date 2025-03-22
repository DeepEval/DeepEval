import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    conv_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1)
    conv_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_2)
    flatten_output = Flatten()(conv_3)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model