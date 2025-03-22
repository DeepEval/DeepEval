from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)
    
    pool = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)

    conv_3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_3_2_1 = Conv2D(filters=32, kernel_size=(1, 3), strides=(2, 2), padding='same', activation='relu')(conv_3_1)
    conv_3_2_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(2, 2), padding='same', activation='relu')(conv_3_1)
    path3 = Concatenate()([conv_3_2_1, conv_3_2_2])

    conv_4_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_4_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_4_1)
    conv_4_3_1 = Conv2D(filters=32, kernel_size=(1, 3), strides=(2, 2), padding='same', activation='relu')(conv_4_2)
    conv_4_3_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(2, 2), padding='same', activation='relu')(conv_4_2)
    path4 = Concatenate()([conv_4_3_1, conv_4_3_2])

    concated = Concatenate()([path1, path2, path3, path4])
    output_layer = Dense(units=10, activation='softmax')(concated)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

