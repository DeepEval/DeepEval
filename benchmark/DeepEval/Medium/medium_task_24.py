from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Concatenate, Dense 

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
        
    conv_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv_1)
    path1 = Dropout(rate=0.5)(conv_2)
    
    conv_3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_3_2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv_3_1)
    conv_3_3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv_3_2)
    conv_3_4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv_3_3)
    path2 = Dropout(rate=0.5)(conv_3_4)
    
    pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    path3 = Dropout(rate=0.5)(pool)

    concated = Concatenate()([path1, path2, path3])
    dense_1 = Dense(units=128, activation='relu')(concated)
    dense_2 = Dense(units=128, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
    