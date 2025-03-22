import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model(): 

    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        batch_norm = BatchNormalization()(input_tensor)
        relu = keras.layers.ReLU()(batch_norm)
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        return conv
    
    pathway1 = block(input_tensor=input_layer)
    pathway1 = block(pathway1)
    pathway1 = block(pathway1)
    
    pathway2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    pathway2 = block(input_tensor=pathway2)
    pathway2 = block(pathway2)
    pathway2 = block(pathway2)
    
    concat = Concatenate()([pathway1, pathway2])
    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model