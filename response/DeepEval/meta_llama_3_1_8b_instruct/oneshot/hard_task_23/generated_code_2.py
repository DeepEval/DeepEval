import keras
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Conv2DTranspose, BatchNormalization, Flatten, Dense
from keras import backend as K

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch2)
    
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3)

    concat_layer = Concatenate()([branch1, branch2, branch3])
    conv_refine = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)
    batch_norm = BatchNormalization()(conv_refine)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model