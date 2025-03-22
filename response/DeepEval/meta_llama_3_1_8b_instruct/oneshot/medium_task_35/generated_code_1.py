import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, Add
from keras.layers import Dropout, Conv2DTranspose, LeakyReLU
from keras.optimizers import Adam
from keras.initializers import he_normal

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Downsampling stage 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = BatchNormalization()(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    # Downsampling stage 2
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    conv2 = BatchNormalization()(conv2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Upsampling stage 1
    upconv1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(maxpool2)
    upconv1 = BatchNormalization()(upconv1)
    upconv1 = LeakyReLU(alpha=0.2)(upconv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    conv3 = BatchNormalization()(conv3)
    skip1 = Add()([upconv1, conv3])
    drop1 = Dropout(0.2)(skip1)

    # Upsampling stage 2
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(drop1)
    upconv2 = BatchNormalization()(upconv2)
    upconv2 = LeakyReLU(alpha=0.2)(upconv2)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop1)
    conv4 = BatchNormalization()(conv4)
    skip2 = Add()([upconv2, conv4])
    drop2 = Dropout(0.2)(skip2)

    conv5 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='tanh')(drop2)

    model = keras.Model(inputs=input_layer, outputs=conv5)

    return model