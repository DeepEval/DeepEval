import keras
from keras.layers import Input, Conv2D, concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Concatenate along the channel dimension
    concat_layer = concatenate([conv3, conv2, conv1], axis=3)
    
    flatten_layer = Flatten()(concat_layer)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model