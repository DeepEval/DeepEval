from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(227, 227, 3))
    conv_1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),padding='valid',activation='relu')(input_layer)
    pool_1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(conv_1)
    conv_2 = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(pool_1)
    pool_2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(conv_2)

    conv_3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(pool_2)
    conv_4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
    pool_3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(conv_5)

    flatten = Flatten()(pool_3)
    dense_1 = Dense(units=4096, activation='relu')(flatten)
    dropout1 = Dropout(rate=0.5)(dense_1)
    dense_2 = Dense(units=4096, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(dense_2)
    dense_3 = Dense(units=1000, activation='softmax')(dropout2)
    output_layer = dense_3

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model