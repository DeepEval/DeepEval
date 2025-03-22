from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)

    pool2_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)  
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2_2)
    path2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv2_2) 

    pool2_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)  
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2_3)
    path3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv2_3)  

    concat = Concatenate()([path1, path2, path3])

    final_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    output_layer = Dense(units=10, activation='softmax')(final_conv)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

