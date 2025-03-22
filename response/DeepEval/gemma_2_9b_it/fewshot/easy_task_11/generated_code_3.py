import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    pool_layer = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(input_layer)
    
    conv_layer = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(pool_layer)
    
    flatten_layer = Flatten()(conv_layer)
    
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout_layer = Dropout(rate=0.2)(dense1)
    dense2 = Dense(units=10, activation='softmax')(dropout_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model