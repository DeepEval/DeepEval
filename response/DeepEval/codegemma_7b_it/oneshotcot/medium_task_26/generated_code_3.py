import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 64))
    
    # Compress input channels
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Expand features through parallel convolutions
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Concatenate results
    concat = Concatenate()([path1, path2])
    
    # Flatten and pass through fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model