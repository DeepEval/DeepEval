import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  

    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    concat2 = Concatenate(axis=3)([conv1, conv2]) 

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat2)
    concat3 = Concatenate(axis=3)([concat2, conv3]) 

    flatten_layer = Flatten()(concat3)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1) 

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model