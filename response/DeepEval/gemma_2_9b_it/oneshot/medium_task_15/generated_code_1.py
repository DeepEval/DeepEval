import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Concatenate, AveragePooling2D

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        bn = BatchNormalization()(conv)
        
        pool = GlobalAveragePooling2D()(bn)
        dense1 = Dense(units=32, activation='relu')(pool)
        dense2 = Dense(units=32, activation='relu')(dense1)
        reshape_layer = Reshape((32, 1, 1))(dense2)  

        weighted_features = keras.layers.multiply([reshape_layer, conv])
        
        concat = Concatenate()([input_layer, weighted_features])
        
        down_sample = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
        down_sample = AveragePooling2D(pool_size=(2, 2))(down_sample)

        output_layer = Dense(units=10, activation='softmax')(down_sample)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model