import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.regularizers import l2
import numpy as np

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = conv1
    
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    path2 = conv2
    
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv4 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
    path3 = Concatenate()([conv4, conv5])
    
    conv6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv7 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv6)
    conv8 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv6)
    path4 = Concatenate()([conv7, conv8])
    
    concat_output = Concatenate()([path1, path2, path3, path4])
    bath_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(bath_norm)
    
    dense1 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(flatten_layer)
    drop_out1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu', kernel_regularizer=l2(0.01))(drop_out1)
    drop_out2 = keras.layers.Dropout(0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(drop_out2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model