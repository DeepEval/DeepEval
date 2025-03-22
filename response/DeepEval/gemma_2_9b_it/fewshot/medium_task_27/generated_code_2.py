import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Add, Flatten, Dense, Activation

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    merged = Add()([branch1, branch2]) 
    
    pool = GlobalAveragePooling2D()(merged)

    dense1 = Dense(units=64, activation='relu')(pool)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model