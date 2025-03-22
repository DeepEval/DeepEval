import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    branch_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1)
    branch_1 = Dropout(0.2)(branch_1)
    
    branch_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_2 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Dropout(0.2)(branch_2)
    
    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    branch_3 = Dropout(0.2)(branch_3)
    
    concat_layer = Concatenate()([branch_1, branch_2, branch_3])
    
    flatten_layer = Flatten()(concat_layer)
    
    dense_layer_1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer_1 = Dropout(0.5)(dense_layer_1)
    
    dense_layer_2 = Dense(units=128, activation='relu')(dense_layer_1)
    dense_layer_2 = Dropout(0.5)(dense_layer_2)
    
    dense_layer_3 = Dense(units=10, activation='softmax')(dense_layer_2)
    
    model = keras.Model(inputs=input_layer, outputs=dense_layer_3)
    
    return model