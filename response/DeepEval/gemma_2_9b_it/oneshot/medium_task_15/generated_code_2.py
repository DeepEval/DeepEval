import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, Concatenate, AveragePooling2D

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        bn = BatchNormalization()(conv)
        
        pool = GlobalAveragePooling2D()(bn)

        fc1 = Dense(units=32, activation='relu')(pool)
        fc2 = Dense(units=32, activation='relu')(fc1)

        reshape_layer = Reshape((32, 1, 1))(fc2) 
        weighted_features = reshape_layer * bn 

        concat = Concatenate()([input_layer, weighted_features])

        conv_down = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
        avg_pool = AveragePooling2D(pool_size=(4, 4))(conv_down) 

        output_layer = Dense(units=10, activation='softmax')(avg_pool)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model