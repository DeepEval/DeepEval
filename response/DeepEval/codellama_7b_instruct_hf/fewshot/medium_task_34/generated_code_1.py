import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv3)
    dropout = Dropout(rate=0.2)(maxpool3)
    up1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(dropout)
    up2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(up1)
    up3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(up2)
    convt1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up3)
    convt2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(convt1)
    convt3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(convt2)
    output = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(convt3)
    model = keras.Model(inputs=input_layer, outputs=output)
    return model