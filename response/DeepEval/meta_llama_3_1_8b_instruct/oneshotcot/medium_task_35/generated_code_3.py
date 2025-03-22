import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    def encoder_block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)

        return path2

    encoder_output = encoder_block(max_pooling1)
    encoder_output = encoder_block(encoder_output)

    def decoder_block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(interpolation='nearest')(input_tensor))
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Concatenate()([path1, path2])
        path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        dropout = Dropout(0.2)(path4)
        return dropout

    decoder_output = decoder_block(encoder_output)
    decoder_output = decoder_block(decoder_output)
    conv_last = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(decoder_output)
    conv_last = UpSampling2D(interpolation='nearest')(conv_last)

    model = keras.Model(inputs=input_layer, outputs=conv_last)

    return model