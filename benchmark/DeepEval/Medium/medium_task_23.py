from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
        
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)
    
    conv_2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_2_2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv_2_1)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(2, 2), padding='same', activation='relu')(conv_2_2)

    conv_3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_3_2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv_3_1)
    conv_3_3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv_3_2)
    conv_3_4 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv_3_3)
    path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(2, 2), padding='same', activation='relu')(conv_3_4)

    pool = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)

    concated = Concatenate()([path1, path2, path3, path4])
    flatten_output = Flatten()(concated)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
