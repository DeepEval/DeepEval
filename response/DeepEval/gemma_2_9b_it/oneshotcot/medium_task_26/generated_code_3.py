import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 64))
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        
        concat_layer = Concatenate()([path1, path2])
        
        flatten_layer = Flatten()(concat_layer)
        dense1 = Dense(units=128, activation='relu')(flatten_layer)
        output_layer = Dense(units=10, activation='softmax')(dense1)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model